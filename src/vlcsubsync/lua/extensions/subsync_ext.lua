--[==========================================================================[
 subsync_ext.lua -- VLC SubSync companion extension (View > SubSync)

 Menu: "Sync subtitles now", "Sync now (exhaustive)", "Auto-sync: ON/OFF"
 (toggle), "Status…", "Load synced result" (only while a fallback job exists),
 "Experimental: no extra track (live delay): ON/OFF" (toggle, writes
 sync_mode=delay|track; the intf then corrects the original track through
 spu-delay instead of adding a synced track, see DESIGN.md "Delay mode"),
 "Use cached results: ON/OFF" (toggle, writes cache=on|off, read by the
 helper; absent = its config.ini, reported in <q>/heartbeat) and "Delete
 cached results" (writes <q>/clear_cache; the helper deletes its result
 cache and that file). Both are for debugging.

 It talks to the interface script (lua/intf/subsync.lua) only through
 <q>/control (written here: auto=1|0, sync_now=<counter>, plus
 sync_now_mode=exhaustive for the exhaustive item) and <q>/intf_state
 (read here), where <q> = vlc.config.userdatadir().."/subsync".
 "Sync now (exhaustive)" asks the helper to transcribe the whole file in
 consecutive 30 s windows (request key mode=exhaustive): more accurate on
 hard files, but it can take minutes (much longer on CPU).
 If the interface script is not running (intf_state missing/stale), "Sync
 subtitles now" falls back to writing the request itself; because extensions
 cannot run a loop, the result is loaded when the user clicks "Load synced
 result" (menu or dialog button), or opportunistically on the next
 input_changed event if the job is done by then.

 VLC 3.0.x extension API notes (modules/lua/extension.c, extension_thread.c,
 libs/dialog.c on the 3.0.x branch):
 * descriptor() returns {title, version, author, shortdesc, description,
   capabilities}; capabilities used here: "menu" (menu()/trigger_menu(id))
   and "input-listener" (input_changed()). close() is called when the user
   closes a dialog window. Hooks run in the extension's own thread.
 * menu() must return {[id]=label}; ids are 16-bit integers.
 * vlc.misc (mdate/mwait) is NOT available to extensions -> no sleeping or
   polling loops here.
 * vlc.dialog(title) -> d:add_label/add_html/add_button(text, cb, col, row,
   col_span, row_span), d:show(), d:update(), d:delete(), w:set_text().
 * vlc.object.input(), vlc.input.item(), vlc.var.get/get_list,
   vlc.input.add_subtitle(path, autoselect), vlc.strings.make_path and
   vlc.osd.message are available to extensions as well.
--]==========================================================================]

local E = {}

E.VERSION = "1.0.2"
E.INTF_MAX_AGE = 15      -- intf_state older than this => intf not running
E.HEARTBEAT_MAX_AGE = 10

-- VLC lists menu entries in id order.
local MENU_SYNC, MENU_SYNC_EXH, MENU_AUTO, MENU_STATUS, MENU_LOAD = 1, 2, 3, 4, 5
local MENU_DELAY = 6
local MENU_CACHE, MENU_CLEAR = 7, 8
E.DELAY_LABEL = "Experimental: no extra track (live delay)"
E.EXHAUSTIVE = "exhaustive"

local ST = {
    dialog = nil,
    job = nil,        -- fallback job: {id, media, uri, status_path, loaded}
    ours = {},        -- spu ES ids added by this extension (fallback mode)
    counter = 0,
}
E.state = ST

---------------------------------------------------------------- helpers

local function log_dbg(msg) pcall(vlc.msg.dbg, "[subsync] " .. tostring(msg)) end
local function log_err(msg) pcall(vlc.msg.err, "[subsync] " .. tostring(msg)) end

local function is_windows()
    return package.config:sub(1, 1) == "\\"
end

local function join(dir, name)
    if dir:sub(-1) == "/" or dir:sub(-1) == "\\" then return dir .. name end
    return dir .. (is_windows() and "\\" or "/") .. name
end

local function one_line(s)
    return (tostring(s or ""):gsub("[\1-\31\127]", " "))
end

local function trim(s)
    return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function html_escape(s)
    return (tostring(s or ""):gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"))
end

function E.parse_kv(data)
    local t = {}
    if not data then return t end
    data = data:gsub("^\239\187\191", "")
    for line in (data .. "\n"):gmatch("([^\n]*)\n") do
        line = line:gsub("\r$", "")
        if not line:match("^%s*[#;]") then
            local k, v = line:match("^%s*([%w_%.%-]+)%s*=(.*)$")
            if k then t[k] = trim(v) end
        end
    end
    return t
end

function E.read_kv(path)
    local f = io.open(path, "rb")
    if not f then return nil end
    local data = f:read("*a")
    f:close()
    return E.parse_kv(data)
end

function E.write_kv(path, kv)
    local lines = {}
    for _, pair in ipairs(kv) do
        lines[#lines + 1] = pair[1] .. "=" .. one_line(pair[2])
    end
    local tmp = path .. ".tmp"
    local f, err = io.open(tmp, "wb")
    if not f then return false, tostring(err) end
    f:write(table.concat(lines, "\n") .. "\n")
    f:close()
    local ok, rerr = os.rename(tmp, path)
    if not ok then
        os.remove(path)
        ok, rerr = os.rename(tmp, path)
    end
    if not ok then
        os.remove(tmp)
        return false, tostring(rerr)
    end
    return true
end

local function mkdir(path)
    if vlc.io and type(vlc.io.mkdir) == "function" then
        pcall(vlc.io.mkdir, path, "0700")
    end
end

function E.queue_dir()
    return join(vlc.config.userdatadir(), "subsync")
end

function E.ensure_dirs()
    local q = E.queue_dir()
    mkdir(q)
    mkdir(join(q, "requests"))
    mkdir(join(q, "jobs"))
    mkdir(join(q, "out"))
    return q
end

function E.uri_to_path(uri)
    if type(uri) ~= "string" or not uri:lower():match("^file://") then return nil end
    if vlc.strings and type(vlc.strings.make_path) == "function" then
        local ok, p = pcall(vlc.strings.make_path, uri)
        if ok and type(p) == "string" and p ~= "" then return p end
    end
    local host, path = uri:sub(8):match("^([^/]*)(/.*)$")
    if not path then return nil end
    path = path:gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end)
    if host ~= "" and host:lower() ~= "localhost" then path = "//" .. host .. path end
    if is_windows() then
        if path:match("^/%a:") then path = path:sub(2) end
        path = path:gsub("/", "\\")
    end
    return path
end

local function osd(text)
    if vlc.osd and type(vlc.osd.message) == "function" then
        pcall(vlc.osd.message, text, 1, "top-right", 3000000)
    end
end

---------------------------------------------------------------- shared files

function E.read_control()
    return E.read_kv(join(E.queue_dir(), "control")) or {}
end

function E.auto_enabled()
    local a = E.read_control().auto
    return not (a == "0" or a == "off" or a == "false")
end

-- "delay" / "track" if the toggle was used, else nil (the helper's config decides)
function E.control_sync_mode(c)
    local m = ((c or E.read_control()).sync_mode or ""):lower()
    if m == "delay" or m == "track" then return m end
    return nil
end

-- Live delay mode in effect: the toggle's choice, else what the intf reports
-- (it knows the helper's configured sync_mode from the last result).
function E.delay_enabled()
    local m = E.control_sync_mode()
    if m then return m == "delay" end
    local st = E.read_kv(join(E.queue_dir(), "intf_state"))
    return st ~= nil and st.sync_mode == "delay"
end

-- "on" / "off" if the cache toggle was used, else nil (the helper's config decides).
-- Accepts what the helper accepts (config.parse_bool).
local CACHE_VALUES = { on = "on", ["1"] = "on", ["true"] = "on", yes = "on",
    off = "off", ["0"] = "off", ["false"] = "off", no = "off" }
function E.control_cache(c)
    return CACHE_VALUES[trim((c or E.read_control()).cache or ""):lower()]
end

-- Result cache in effect: the toggle's choice, else the helper's config.ini (its
-- heartbeat reports it), else on.
function E.cache_enabled()
    local v = E.control_cache()
    if v then return v == "on" end
    local hb = E.read_kv(join(E.queue_dir(), "heartbeat"))
    return not (hb and hb.cache == "off")
end

-- `mode` (optional) applies to this sync_now increment ("" / nil = default mode).
-- `sync_mode` "delay"/"track" and `cache` "on"/"off" set their toggles; nil keeps
-- the current value.
function E.write_control(auto, sync_now, mode, sync_mode, cache)
    E.ensure_dirs()
    local kv = {
        { "auto", auto and 1 or 0 },
        { "sync_now", sync_now },
    }
    if mode and mode ~= "" then kv[#kv + 1] = { "sync_now_mode", mode } end
    sync_mode = sync_mode or E.control_sync_mode()
    if sync_mode then kv[#kv + 1] = { "sync_mode", sync_mode } end
    cache = cache or E.control_cache()
    if cache then kv[#kv + 1] = { "cache", cache } end
    return E.write_kv(join(E.queue_dir(), "control"), kv)
end

-- Returns intf_state table (or nil) and whether the intf looks alive.
function E.intf_status()
    local st = E.read_kv(join(E.queue_dir(), "intf_state"))
    if not st then return nil, false end
    local t = tonumber(st.time)
    local alive = t ~= nil and os.time() - t <= E.INTF_MAX_AGE and st.state ~= "stopped"
    return st, alive
end

function E.daemon_status()
    local hb = E.read_kv(join(E.queue_dir(), "heartbeat"))
    local t = hb and tonumber(hb.time)
    if not t then return hb, false, nil end
    local age = os.time() - t
    return hb, age <= E.HEARTBEAT_MAX_AGE, age
end

---------------------------------------------------------------- fallback sync

local function ordinal(ids, labels, id, skip)
    local n = 0
    for i, v in ipairs(ids) do
        if v ~= -1 and not (skip and skip[v]) then
            if v == id then return n, labels[i] or "" end
            n = n + 1
        end
    end
    return nil
end

local function es_list(input, var)
    local ok, ids, labels = pcall(vlc.var.get_list, input, var)
    if not ok or type(ids) ~= "table" then return {}, {} end
    return ids, (type(labels) == "table") and labels or {}
end

-- Current selection of the playing input, or nil + reason.
function E.current_selection()
    local input = vlc.object.input()
    if not input then return nil, "Nothing is playing." end
    local item = vlc.input.item()
    local uri = item and item:uri()
    local media = E.uri_to_path(uri)
    if not media then return nil, "Only local files are supported." end
    if uri ~= ST.ours_uri then
        ST.ours, ST.ours_uri = {}, uri
    end
    local aids, alabels = es_list(input, "audio-es")
    local sids, slabels = es_list(input, "spu-es")
    local audio = tonumber(vlc.var.get(input, "audio-es")) or -1
    local spu = tonumber(vlc.var.get(input, "spu-es")) or -1
    if spu == -1 then return nil, "Select a subtitle track first." end
    if ST.ours[spu] then return nil, "The synced track is already selected." end
    local a, alabel = ordinal(aids, alabels, audio)
    local s, slabel = ordinal(sids, slabels, spu, ST.ours)
    if not a then return nil, "No audio track selected." end
    if not s then return nil, "Subtitle track not found." end
    return { media = media, uri = uri, audio = a, audio_label = alabel,
             sub = s, sub_label = slabel, spu_ids = sids }
end

function E.fallback_sync(mode)
    local sel, why = E.current_selection()
    if not sel then return false, why end
    local q = E.ensure_dirs()
    ST.counter = ST.counter + 1
    local id = tostring(os.time()) .. "_e" .. ST.counter
    local kv = {
        { "version", 1 },
        { "id", id },
        { "media", sel.media },
        { "audio_index", sel.audio },
        { "audio_label", sel.audio_label },
        { "sub_index", sel.sub },
        { "sub_label", sel.sub_label },
        { "sub_path", "" },
        { "force", 0 },
    }
    if mode and mode ~= "" then kv[#kv + 1] = { "mode", mode } end
    local ok, err = E.write_kv(join(join(q, "requests"), id .. ".req"), kv)
    if not ok then return false, "Cannot write request: " .. err end
    ST.job = { id = id, media = sel.media, uri = sel.uri, mode = mode,
               status_path = join(join(q, "jobs"), id .. ".status") }
    log_dbg("fallback request " .. id)
    return true
end

-- Check the fallback job; load the result if done. Returns a status line.
function E.check_job(load)
    local job = ST.job
    if not job then return "No sync job." end
    if job.loaded then return "Synced subtitles loaded." end
    local st = E.read_kv(job.status_path)
    if not st then
        local _, alive = E.daemon_status()
        if alive then return "Waiting for the SubSync helper…" end
        return "Waiting for the SubSync helper (it does not seem to be running)."
    end
    local state = st.state or "queued"
    if state == "queued" or state == "running" then
        local p = math.floor((tonumber(st.progress) or 0) * 100 + 0.5)
        local slow = (job.mode == E.EXHAUSTIVE) and " (exhaustive, may take a while)" or ""
        return string.format("Syncing%s… %d%% %s", slow, p, st.message or "")
    elseif state == "error" then
        ST.job = nil
        return "Error: " .. (st.message or "unknown")
    elseif state == "done" then
        if st.applied ~= "1" or not st.output or st.output == "" then
            ST.job = nil
            return "Not synced (" .. (st.message or "") .. "); keeping original timing."
        end
        if not load then return "Done: " .. (st.message or "") .. " – click “Load synced result”." end
        local input = vlc.object.input()
        local item = input and vlc.input.item()
        if not item or item:uri() ~= job.uri then
            return "Done, but the media is no longer playing."
        end
        local before = {}
        for _, v in ipairs((es_list(input, "spu-es"))) do before[v] = true end
        ST.pending_before = before
        ST.ours_uri = job.uri
        local ok, err = pcall(vlc.input.add_subtitle, st.output, true)
        if not ok then return "Could not load subtitles: " .. tostring(err) end
        job.loaded = true
        osd("Subtitles synced: " .. (st.message or ""))
        return "Synced subtitles loaded: " .. (st.message or "")
    end
    return "Unknown state " .. tostring(state)
end

-- Remember tracks this extension added (so they are excluded from ordinals).
function E.collect_ours()
    if not ST.pending_before then return end
    local input = vlc.object.input()
    if not input then return end
    local ids = es_list(input, "spu-es")
    for _, v in ipairs(ids) do
        if v ~= -1 and not ST.pending_before[v] then
            ST.ours[v] = true
            ST.pending_before = nil
        end
    end
end

---------------------------------------------------------------- dialog

local function close_dialog()
    if ST.dialog then
        pcall(function() ST.dialog:delete() end)
        ST.dialog = nil
    end
end

function E.status_html()
    local parts = {}
    local hb, d_alive, age = E.daemon_status()
    if d_alive then
        parts[#parts + 1] = string.format("<b>Helper daemon:</b> running (v%s, heartbeat %ds ago)",
            html_escape(hb.version or "?"), age)
    elseif hb then
        parts[#parts + 1] = string.format("<b>Helper daemon:</b> <font color=red>not running</font>"
            .. " (last heartbeat %ss ago). Start it with <tt>vlc-subsync serve</tt>"
            .. " or run <tt>vlc-subsync doctor</tt>.", tostring(age or "?"))
    else
        parts[#parts + 1] = "<b>Helper daemon:</b> <font color=red>never seen</font>."
            .. " Install it and run <tt>vlc-subsync setup</tt>."
    end
    local st, i_alive = E.intf_status()
    if i_alive then
        parts[#parts + 1] = "<b>Automatic mode:</b> interface script running"
        parts[#parts + 1] = "<b>State:</b> " .. html_escape(st.state or "")
            .. ((st.message and st.message ~= "") and (" – " .. html_escape(st.message)) or "")
        if st.last_result and st.last_result ~= "" then
            parts[#parts + 1] = "<b>Last result:</b> " .. html_escape(st.last_result)
        end
    else
        parts[#parts + 1] = "<b>Automatic mode:</b> <font color=red>the SubSync interface script"
            .. " is not running.</font> Run <tt>vlc-subsync setup</tt> and restart VLC to enable"
            .. " automatic syncing. “Sync subtitles now” still works from this menu."
        if st and st.last_result and st.last_result ~= "" then
            parts[#parts + 1] = "<b>Last result:</b> " .. html_escape(st.last_result)
        end
    end
    parts[#parts + 1] = "<b>Auto-sync:</b> " .. (E.auto_enabled() and "ON" or "OFF")
    if E.delay_enabled() then
        parts[#parts + 1] = "<b>Mode:</b> live delay (experimental, no extra track)"
    end
    if not E.cache_enabled() then
        parts[#parts + 1] = "<b>Cached results:</b> not used (every sync runs again)"
    end
    if ST.job then
        parts[#parts + 1] = "<b>Manual job:</b> " .. html_escape(E.check_job(false))
    end
    return table.concat(parts, "<br>")
end

function E.show_status(extra)
    close_dialog()
    local d = vlc.dialog("SubSync status")
    ST.dialog = d
    local html = E.status_html()
    if extra then html = "<b>" .. html_escape(extra) .. "</b><br><br>" .. html end
    ST.status_widget = d:add_html(html, 1, 1, 3, 1)
    d:add_button("Refresh", function()
        E.collect_ours()
        ST.status_widget:set_text(E.status_html())
        pcall(function() d:update() end)
    end, 1, 2, 1, 1)
    if ST.job then
        d:add_button("Load synced result", function()
            local msg = E.check_job(true)
            ST.status_widget:set_text("<b>" .. html_escape(msg) .. "</b><br><br>" .. E.status_html())
            pcall(function() d:update() end)
        end, 2, 2, 1, 1)
    end
    d:add_button("Close", close_dialog, 3, 2, 1, 1)
    d:show()
    return d
end

---------------------------------------------------------------- actions

-- True if the running intf (its intf_state `st`) understands sync_now_mode=`mode`.
-- Interface scripts from before sync modes do not write the `modes` key.
function E.intf_supports(st, mode)
    local modes = st and st.modes
    if not modes then return false end
    for m in modes:gmatch("[^,%s]+") do
        if m == mode then return true end
    end
    return false
end

-- `mode`: nil/"" = the helper's configured mode, or "exhaustive".
function E.sync_now(mode)
    local exhaustive = mode == E.EXHAUSTIVE
    local st, i_alive = E.intf_status()
    if i_alive and mode and mode ~= "" and not E.intf_supports(st, mode) then
        -- An interface script from before sync modes (still running after an
        -- upgrade, until VLC restarts) would ignore sync_now_mode and run a normal
        -- sync; a request of our own would race with it over the loaded track.
        E.show_status("The running SubSync interface script is from an older version"
            .. " and cannot run a " .. mode .. " sync. Restart VLC, then try again.")
        return
    end
    if i_alive then
        local c = E.read_control()
        local n = (tonumber(c.sync_now) or 0) + 1
        local auto = c.auto ~= "0"
        local ok, err = E.write_control(auto, n, mode)
        if not ok then
            E.show_status("Cannot write control file: " .. tostring(err))
            return
        end
        osd(exhaustive and "SubSync: exhaustive sync requested (may take a while)"
            or "SubSync: sync requested")
        return
    end
    local ok, why = E.fallback_sync(mode)
    if ok then
        osd(exhaustive and "Syncing subtitles (exhaustive, may take a while)…"
            or "Syncing subtitles…")
        E.show_status("Sync requested. The SubSync interface script is not running, so"
            .. " click “Load synced result” when the job is done (or reopen this dialog"
            .. " from the menu).")
    else
        E.show_status(why)
    end
end

function E.toggle_auto()
    local c = E.read_control()
    local auto = c.auto ~= "0"
    local ok, err = E.write_control(not auto, tonumber(c.sync_now) or 0)
    if not ok then
        E.show_status("Cannot write control file: " .. tostring(err))
        return
    end
    local _, i_alive = E.intf_status()
    if i_alive then
        osd("SubSync auto-sync " .. ((not auto) and "ON" or "OFF"))
    else
        E.show_status("Auto-sync is now " .. ((not auto) and "ON" or "OFF")
            .. ", but the interface script is not running.")
    end
end

-- Experimental live delay mode on/off (sync_mode=delay|track in <q>/control).
function E.toggle_delay()
    local c = E.read_control()
    local on = not E.delay_enabled()
    local st, i_alive = E.intf_status()
    if on and i_alive and not (st and st.sync_modes and st.sync_modes:find("delay", 1, true)) then
        E.show_status("The running SubSync interface script is from an older version"
            .. " and has no live delay mode. Restart VLC, then try again.")
        return
    end
    local ok, err = E.write_control(c.auto ~= "0", tonumber(c.sync_now) or 0, nil,
        on and "delay" or "track")
    if not ok then
        E.show_status("Cannot write control file: " .. tostring(err))
        return
    end
    if i_alive then
        osd("SubSync live delay (experimental) " .. (on and "ON" or "OFF"))
    else
        E.show_status("Live delay mode is now " .. (on and "ON" or "OFF")
            .. ", but the interface script is not running.")
    end
end

-- The running helper (alive, heartbeat `hb`) knows the cache toggle and the
-- delete command: helpers from before them write no `cache` key.
local function helper_too_old(hb, d_alive)
    return d_alive and (hb.cache == nil or hb.cache == "")
end

local OLD_HELPER = "The running SubSync helper is from an older version and cannot do"
    .. " this. Restart VLC (the helper restarts with it), then try again."

-- Debugging: stop (or resume) reusing and storing results (cache=on|off in <q>/control).
function E.toggle_cache()
    local hb, d_alive = E.daemon_status()
    if helper_too_old(hb, d_alive) then
        E.show_status(OLD_HELPER)
        return
    end
    local c = E.read_control()
    local on = not E.cache_enabled()
    local ok, err = E.write_control(c.auto ~= "0", tonumber(c.sync_now) or 0, nil, nil,
        on and "on" or "off")
    if not ok then
        E.show_status("Cannot write control file: " .. tostring(err))
        return
    end
    if d_alive then
        osd("SubSync cached results " .. (on and "ON" or "OFF"))
    else
        E.show_status("Cached results are now " .. (on and "ON" or "OFF")
            .. ". The helper is not running; it uses this setting once it starts.")
    end
end

-- Debugging: ask the helper to delete every stored result (<q>/clear_cache).
function E.clear_cache()
    local hb, d_alive = E.daemon_status()
    if helper_too_old(hb, d_alive) then
        E.show_status(OLD_HELPER)
        return
    end
    E.ensure_dirs()
    local ok, err = E.write_kv(join(E.queue_dir(), "clear_cache"), { { "time", os.time() } })
    if not ok then
        E.show_status("Cannot ask the helper to delete cached results: " .. tostring(err))
        return
    end
    if d_alive then
        osd("SubSync: deleting cached results")
    else
        E.show_status("The helper is not running. It deletes the cached results when"
            .. " it starts, or run vlc-subsync clear-cache.")
    end
end

---------------------------------------------------------------- VLC hooks

-- PNG shown in Tools > Plugins and extensions > Active Extensions
-- ICON BEGIN (generated by scripts/embed_ext_icon.py)
local ICON_PNG =
    "\137PNG\013\010\026\010\000\000\000\013IHDR\000\000\000@\000\000\000@\008\006\000\000\000\170iq\222\000" ..
    "\000\012\009IDATx\218\229\155m\140\092\213y\199\127\2079\231\222;;\251\226\005\175\247\197\139qc\012Q\028" ..
    "\194\139q\176(\133BI\160\130\016\145\022B\1654j)|H\1656\138R!UQ\162*\149\242)U\211 P\233K\148V\249\018\161" ..
    "T*\193\144\146\164\149\021\133\170\164\142M\169\193!\137\169\215\235\151\245\218\235]\239\206\206\206\204" ..
    "\189\231<\253pg_l\239\174gvg\140\213\030\233jw\238\204\189\247<\255\231\255\188\158s\133\149\135\001\194" ..
    "\252\135\190\161-w\161\225c\168\220)\176\013\213\141\136\196\092\009C\181\138\200\024\202A\132W\013\246" ..
    "\165S\167F\142\212\191\181\128_\233RY\225\252\252Ev\211\224\150O\161\250\135\192n\017#\160\168*W\222\016D" ..
    "\242\191\170z\022\228;\193f\1279q\226\196;K\228\212F\000\176\128\031\024\024\222\029\196|\221\136\236V]" ..
    "\016\218\215\175\145U\192{\207x\176D@c\140!\168\206\162\250\229\211\167\142}u9V/\007\128\005|\127\255\230" ..
    "\207\168\152\175\139\152X5\204\011mZ\175\179\021\212\210\0260<\224\1401\132\016\190[\171\216\223;wnd\234B" ..
    "\016\150\002\224\128\172\175\127\248\011\198\216\175\168\006\173\255\208\182C]V\192\235\197\255\183\003" ..
    "\0081\198i\008\251\172d\015\142\141\141\157^\010\130Y\162\249\172\191\127\243g\140\181_Q\013\217\146\243m" ..
    "\025\147\213E;\154\172\182\2091\128\211\016R1f\151\015\238\165\129\129\129\206\165\202\183\243\180\2238" ..
    "\184e\151 \255T7v\219N\027O\003<}\171\242\023w+O\236P\174\138\225\223\199\004\219>\175bQM\197\154k5\200" ..
    "\182\242\236\204w\2342\234\188\160\174\171\179\231E\017\025\174\211\198\180\139\246\231j\240\249[\148/\223" ..
    "\171\244\197\208\215\001\247l\135\242\028\252\2191\161\232\218\226\019\150\128`o.v\246\140\148g\167\015" ..
    "\000\214\000ac\255\240\227b\204NU\205\218I{\175\208\025\193\239\092\175\248*di~\248j~\1743j\155/X\000A\131" ..
    "\006U\253\234\208\208P\031\016L\029\133\207\213\169/\2374\198<g\201\133\020Y\018O\235N\2402\164\023\0064Xk" ..
    "\250\178`>\011\168\217\180i\203\029\136\220Z\143\243m\209\1901\006DpV(g\240\205C\130\137\192\218\2520Q~" ..
    "\174\156\177\162\031\016i\153nL\008\170\192\147\027\182n\237\181\197\238\238\207\0261w\214\195B\203m_U\153" ..
    "\155\171\2243O\230\003]\137\227\245Sp|\010><\008\229\012\254t\175\240\015?\019\186\162\011\178\148%\247" ..
    "\240\222\231@\182\134\140*bz\092\234\2232 w\092\034-^\253n\034+NLD\200\178\140?\251\210\211\236y\233\005" ..
    "\182o\127\031\149j\149\158Xx\238\128\176\239$\236;\009\207\030\016:Wp~!\004\2268\166w\195\006B\008-\205" ..
    "\026\021\249\184\001\221V\183=Y\139\240\181Z\202\204L\137\011\025:\255\221\224\224\000O\254\254\167\184" ..
    "\235\174{x\236\183>\206\236l\0251\134$\001g\242#I\206\023^D\022(o\173\229\142\221\187\248\232G\238ahh\144," ..
    "\203Za\014FATt\167\017\228\234\250\227\165y\225k\012o\030\228\215\239\254U\178\204/\011\1801\134ry\014\239" ..
    "k \139\182\236u1y_\234\249U\149Z\173F\154f\132\016H\146\132\238\238.\156\179l\188\250*B\208U\205\173Al\004" ..
    "UP\134\013\016\175E\243A\1498\142\249\198\223>\195+/\191\196'\031}\132\153\153\018\2069\172\181\231\153" ..
    "\1331\006kmC\254\034I\018v\221v\011[\175\189\166Ny%\132\128\170\174h\002\243\213\169\181\022\239C3r\020" ..
    "\154\246*\243\194eY\198\134\013=\092\187e3\170\158\155>\180\131\016\002\181Z\202\212\212\0203\165Rs\160" ..
    "\026!M3\182n\221\194\141;>\192\2057}\144$Ir\141/\163\214\165L\016\017n\223\181\147\143\222w\015\189\189" ..
    "\027\240\2227l&\166Y\143~vr\146r\185\140\017\193{O\154f\136@\154\166x\159r\237\150a\190\245\143\127\199SO" ..
    "\252.\213j\021\211\164\189Zc\168\214\170u\176\013\160\023\217f\008\129\142B\130\212\231P,v00\176\137\222" ..
    "\222\030\006\007\006\154b\129iFxk-O=\241in\255\240mT*\021\172\181\139\206\2029|:\199\227\159\252\004\143>" ..
    "\250\024_\252\194\159\208\223\223O-M\155vZ+\253~>\170\012\013\013\242\155\015\220\199\206[oZH\158\188\247d" ..
    "\153o\186Y\227\026\165\253\228\228$O\254\193\167y\238\217g\025=\242s~\237\222\135H\151\017\2069\135\2475" ..
    "\210ZJ\146\196-\239\030\169*\003\155\250(\020\018\006\250\251\243g\004=/r4\153\0266\162\145\220\230\134" ..
    "\006\250\241\190F\177\216\193UW\245\146\249\139C\210<SD\164m\173\179\016\002!\132\166\168\222\018\031\144e" ..
    "\025\214\154\250\195=\242\030w\197Z\145\029\155\181\160\159\199\219+\173%\216f\000T\021\235\028\198\196" ..
    "\020\139\197U\127\215\238\206q+\205\2034\166u\165X,\242\189W\127\200\161Co\242\242+?`t\2448q\028\019.\152H" ..
    "\028\199\128!\138\162V\021/\023\141Z\154\146$I}n~]\247r\141\210\190PHx\251\237w\184\247\190\135\2434u^\240" ..
    "\250_\013\010b9\240\198\155\128\231\200\200Q\206\156\153\160\191\127SK#\128s\142w\255g\004g-\019g\207R\171" ..
    "\165\020\139\197<\179]\131i\186f\030\158$\009\222{\162\200\161\026\016Y,UC\240\196I'{^\254>\247\254\198" ..
    "\131\140\143\159\166R\169`\140i\138\174\198\152\133\010S\016\244\130\026q\190\006\217\255\198\127c\173\193" ..
    "9GZKQ\148$\142\169\213jMU5\174Y\013\204\011\148'%\1588\142\1776^\248>\142#\246\253\244\000q\028\019\199M" ..
    "\228\001\010\198\008g\206L\144\166)SS\211T\170U\010\133\228\162{\136\008I\178X\194T*U\246\237;\192\134\013" ..
    "=\028\029=F\228\092\195\207uk\161!@\020ELLL\242\181g\158\231\129\007\238\231\219/\2523\029\029\005B\008tv" ..
    "\230\157\231j\181z\2225\171Mj\158\222c\167\198y\245\007{\169\213j\248\0160b\150\189v\233gc\013'O\141s\252" ..
    "\196\024Q\228\1542\003\183V{\156\247\011\207=\255\013\158\251\235\191\1999G\161PX\168\216\230'aLN\211\016" ..
    "\002Q\0205d\002\165R\009\017\193\026CV\239\004]\138M\145s\224hO*\188\218\232\170k{\201\250\225y)\244\217" ..
    "\201I\198\199O\211?\180\149#G\142\230&\212@\234=\127\175J\165\194\193\183~F\127\127\031#G\143\225\156]\149" ..
    "\153M\135\212M\003\215\180-`\139\008\149j\149\027\182_\199\245\215_\199\015\255uoNu#\156\169\192\203\031" ..
    "\203\217\242\208\030C_a\229\150x^\228\132\139\250\012\173\024\1426\014U\165\144$\252\226\151\1359\248\214!" ..
    "\186\187\187\214\148AF\145[\151\150\2233\000\022@($ttt\224\189_\243=\2185\218\014\192b\247\198s%\014\195" ..
    "\255\243\2094\003\140\172\127\253\204J~H\189+\15673Z\179.\215\2140h\004J)\164~}KUV\160ZU\178\160\024\129" ..
    "\180V\163f\004\191f[\207Sfk,\206\217\166@p\205\010\127\247\176\225\183o\016\138\182^\007\173\001\007\145" ..
    "\192\1553\189\244\245\214\2000\220~\2516z\157g=\253\157\224\003G\142\14225u\014\215D*\220P\030`\004\202)" ..
    "\2206h\248\254#\129(\246kYK9?\241\199\162\153\231'\149-<?\179\155\162\164\132u\220\207\024\195\236\236\028" ..
    "{\127\244\026\165R\169\225|\1935\170\253r\006\015n\133(\246\148KB\180\174ud\193\163$&\197{\165R\173b\215" ..
    "\005@n\251\197b\129\225\205\131\188\245\246;$Ic\166\208X?@!\177\240\250\169|\179X\177\016\234\222k\237\016" ..
    "D\162\016\0041B\020ED\162k\006@u\209'\157\155\158\193\152\022\023CA\161+\134\1279\226yz\175\229\137\027" ..
    "\013E\147/\175\201\026\200/\0188<\215\205\182\174\020\235\171LNM\147\217\148\160\235\139\003GFF\025\027;" ..
    "\133sQk}\192\034qa:\133\238\216\016\219evt4\016\202\020p\002Gg-\175<\148\226$\240\208\030\187j-\208hu\234" ..
    "\189\199\185\230\034\187kV{\1891x\013T\210\243\191\017\017\208\198\226p&\224B\134CQ\149z\159?gZ#\005\214J" ..
    "\231\163(j\1279\188ts\227\226\211\013>\203\022\236\249R\182a\005\172\023\144,\223\162\230\028\214\234%\183" ..
    "\135\137\008i\154^T2\175\167fXs-\160K&\149e\025}\027\175\230C7\238 N\226K\010bD\153\246\017\0277\238\007" ..
    "\224\254\143\236\164\167\001\031 \034\204\205\205\241\198\155\00797=\131\179v\221\133\210\186\139!U%\138" ..
    "\028\183\237\188\133\222\222\013dYz\201\240`P4\196\011\246\218\221\213E\151\169]2\010(Joo\015Q\020\177\247" ..
    "G?nI\149\184n\000\242\214X\007\029\029\005\170\213jC\1472(Y0\011k\252Y\150\145\153\172\1610X\241\129\174" ..
    "\174N\146\164\144/\211\175\179A\226\020\173\008RXs^o-\179\179eF\143\029\231\253\215o'\243YC\012H5\174\175" ..
    "\255C\161\144\144\1364\000@\2228=\252\238HK\132\007j\0148\142\200uk\221(9\223*\127\227\191\014255Mww\231%o" ..
    "eQ\1662G\233\131\249.\146\183\015\253\156^\151\225W}|\030ifg\23182rt\189k\147\010\136\162g\157\168\236\023" ..
    "a\155\174ck\252\252d~\241\203\195\013W\131\019\021\152\188&\215\224\190\159\030`c\019y\128sn\221\000\136" ..
    "\136\168\234\187N\208\239\130<F\011\202\241|]\1761\000\092\128\158\216\131\128\139c\226\184q\000Z\224\252" ..
    "\020\004#\188\230\170\029nO\092\201N\128\025\002]\215n\209F'\022\234\217\224+#y#\196\137\018.\207^\225\005" ..
    "7\164\026\002!\188(\000\155\006\134\255\220\024\251\165\016Bv\185\250\132B^a\002\237\220\034\191l.'\034F5" ..
    "\236?}\234\248n\003\1363\225\025\239\195\025\144\139^*j\215P\242\173\243\157\209e\021~\193\155\006\229\175" ..
    "\000o\001[*\149f;\187z\206\1361\143\208\230w\006.\004\2252\011\239E\196\169\234\2543\227\199?G]P\005lyvz" ..
    "\127\177\179\235\253b\236\205\168\166\151\011\132\203\170\249\250\206\220 \242\137\185\210\2441`!\147\008" ..
    "\128q&<\165!\252\135\024\019\001\233\2551\225\189\024cE\245\143'\198F\247\213\021\236\205\210\218\230\228" ..
    "\201\147eg\252\195\026\244?\235 d\151\159\165\173\167}\158\171\024\023\188~q|\252\248\223\212\029\189\231" ..
    "\002\154+`J\165\210l\018\247\189`L\246\001c\204\014U\149:\016W\226\219\162\151\138\182^\1968\2084\168\255" ..
    "\163\137\241\227_\227\130w\137\2372T1\149\202d\165<;\253\237bW\247\172\138\220i\141)\212\129\008K+\225+" ..
    "\212\198\003`DD\1401F\131\190nD\031?=~\226E\150y\145ZV\009\211\000\218\215\183\249\006\177\230\243 \143" ..
    "\137\228\239\022\228\009\203\149g\025yz,\168\134\000\252\004\145\231O\143\141~\139\1974\223\175$\232j\229r" ..
    "\006000\240+\129\232a\148\2511\220H\208!D\146+C\247ZCdB\2250\194k\034\178\231\244\201\209\031\159W\128\174" ..
    "\144\223\252/W\164\190 \228\212\188\010\000\000\000\000IEND\174B`\130"
-- ICON END

function descriptor()
    return {
        title = "SubSync – automatic subtitle sync",
        version = E.VERSION,
        author = "VLC SubSync",
        url = "https://github.com/sergimn/VLCSubtitleSync",
        shortdesc = "SubSync",
        description = "Re-times the selected subtitle track to the selected audio track"
            .. " (constant delay and drift) using the vlc-subsync helper.",
        capabilities = { "menu", "input-listener" },
        icon = ICON_PNG,
    }
end

function activate()
    pcall(E.ensure_dirs)
end

function deactivate()
    close_dialog()
end

function close()
    close_dialog()
end

function menu()
    local m = {}
    m[MENU_SYNC] = "Sync subtitles now"
    m[MENU_SYNC_EXH] = "Sync now (exhaustive)"
    local ok, auto = pcall(E.auto_enabled)
    m[MENU_AUTO] = "Auto-sync: " .. ((not ok or auto) and "ON" or "OFF")
    if ST.job and not ST.job.loaded then m[MENU_LOAD] = "Load synced result" end
    m[MENU_STATUS] = "Status…"
    local okd, delay = pcall(E.delay_enabled)
    m[MENU_DELAY] = E.DELAY_LABEL .. ": " .. ((okd and delay) and "ON" or "OFF")
    local okc, cache = pcall(E.cache_enabled)
    m[MENU_CACHE] = "Use cached results: " .. ((not okc or cache) and "ON" or "OFF")
    m[MENU_CLEAR] = "Delete cached results"
    return m
end

function trigger_menu(id)
    local ok, err = pcall(function()
        if id == MENU_SYNC then
            E.sync_now()
        elseif id == MENU_SYNC_EXH then
            E.sync_now(E.EXHAUSTIVE)
        elseif id == MENU_AUTO then
            E.toggle_auto()
        elseif id == MENU_LOAD then
            E.show_status(E.check_job(true))
        elseif id == MENU_DELAY then
            E.toggle_delay()
        elseif id == MENU_CACHE then
            E.toggle_cache()
        elseif id == MENU_CLEAR then
            E.clear_cache()
        elseif id == MENU_STATUS then
            E.collect_ours()
            E.show_status()
        end
    end)
    if not ok then log_err("menu action failed: " .. tostring(err)) end
end

function input_changed()
    pcall(function()
        E.collect_ours()
        if ST.job and not ST.job.loaded then
            local input = vlc.object.input()
            local item = input and vlc.input.item()
            if item and item:uri() == ST.job.uri then
                local st = E.read_kv(ST.job.status_path)
                if st and st.state == "done" then E.check_job(true) end
            end
        end
    end)
end

function meta_changed()
end

-- plain globals only: VLC runs extensions without rawget (and _G may be absent)
if SUBSYNC_TEST then
    subsync_ext = E
end
