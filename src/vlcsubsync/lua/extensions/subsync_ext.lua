--[==========================================================================[
 subsync_ext.lua -- VLC SubSync companion extension (View > SubSync)

 Menu: "Sync subtitles now", "Auto-sync: ON/OFF" (toggle), "Load synced
 result" (only while a fallback job exists), "Status…".

 It talks to the interface script (lua/intf/subsync.lua) only through
 <q>/control (written here: auto=1|0, sync_now=<counter>) and <q>/intf_state
 (read here), where <q> = vlc.config.userdatadir().."/subsync".
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

E.VERSION = "0.1.0"
E.INTF_MAX_AGE = 15      -- intf_state older than this => intf not running
E.HEARTBEAT_MAX_AGE = 10

local MENU_SYNC, MENU_AUTO, MENU_STATUS, MENU_LOAD = 1, 2, 3, 4

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
    return (tostring(s or ""):gsub("[%c]", " "))
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

function E.write_control(auto, sync_now)
    E.ensure_dirs()
    return E.write_kv(join(E.queue_dir(), "control"), {
        { "auto", auto and 1 or 0 },
        { "sync_now", sync_now },
    })
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

function E.fallback_sync()
    local sel, why = E.current_selection()
    if not sel then return false, why end
    local q = E.ensure_dirs()
    ST.counter = ST.counter + 1
    local id = tostring(os.time()) .. "_e" .. ST.counter
    local ok, err = E.write_kv(join(join(q, "requests"), id .. ".req"), {
        { "version", 1 },
        { "id", id },
        { "media", sel.media },
        { "audio_index", sel.audio },
        { "audio_label", sel.audio_label },
        { "sub_index", sel.sub },
        { "sub_label", sel.sub_label },
        { "sub_path", "" },
        { "force", 0 },
    })
    if not ok then return false, "Cannot write request: " .. err end
    ST.job = { id = id, media = sel.media, uri = sel.uri,
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
        return string.format("Syncing… %d%% %s", p, st.message or "")
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

function E.sync_now()
    local _, i_alive = E.intf_status()
    if i_alive then
        local c = E.read_control()
        local n = (tonumber(c.sync_now) or 0) + 1
        local auto = c.auto ~= "0"
        local ok, err = E.write_control(auto, n)
        if not ok then
            E.show_status("Cannot write control file: " .. tostring(err))
            return
        end
        osd("SubSync: sync requested")
        return
    end
    local ok, why = E.fallback_sync()
    if ok then
        osd("Syncing subtitles…")
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

---------------------------------------------------------------- VLC hooks

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
    local ok, auto = pcall(E.auto_enabled)
    m[MENU_AUTO] = "Auto-sync: " .. ((not ok or auto) and "ON" or "OFF")
    if ST.job and not ST.job.loaded then m[MENU_LOAD] = "Load synced result" end
    m[MENU_STATUS] = "Status…"
    return m
end

function trigger_menu(id)
    local ok, err = pcall(function()
        if id == MENU_SYNC then
            E.sync_now()
        elseif id == MENU_AUTO then
            E.toggle_auto()
        elseif id == MENU_LOAD then
            E.show_status(E.check_job(true))
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

if rawget(_G, "SUBSYNC_TEST") then
    _G.subsync_ext = E
end
