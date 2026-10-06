--[==========================================================================[
 subsync.lua -- VLC SubSync interface script (automatic subtitle re-timing)

 Enabled through vlcrc:  extraintf=luaintf  lua-intf=subsync
 Talks to the `vlc-subsync serve` daemon through a file queue in
 vlc.config.userdatadir().."/subsync" (see DESIGN.md "File protocol").

 VLC 3.0.x Lua API notes (checked against modules/lua/{intf.c,libs/*.c} and
 share/lua/README.txt on the 3.0.x branch):
 * VLC 3 bundles Lua 5.1/5.2/LuaJIT depending on the build -> this file sticks
   to the common subset (no //, no bit ops, no goto, no table.unpack).
 * Interface scripts run in their own thread; luaL_openlibs() is called, so the
   standard io/os/string/table libraries are present next to vlc.io.
 * vlc.misc.mwait(date_us) / vlc.misc.mdate() exist (interfaces only).
   VLC 3 has NO vlc.misc.should_die(): when the interface is closed, VLC
   interrupts the thread and mwait() raises the Lua error "Interrupted." (and
   keeps raising it). So mwait is called outside the per-tick pcall and an
   error from it ends the loop. should_die() is still honoured if present
   (VLC 2.x / possible future versions).
 * vlc.object.input() returns the current input object or nil (3.0; it was
   removed in VLC 4 where the player API replaced it -> we then just idle).
 * vlc.var.get_list(input, "spu-es"|"audio-es") returns two arrays: the ES ids
   (integers, -1 = "Disable") and their display labels
   ("Disable", "Track 1 - [English]", ...). vlc.var.get/set(input, "spu-es")
   read/select the current ES id (-1 = disabled).
 * vlc.input.add_subtitle(path, autoselect) takes a local *path* in 3.0 (it
   calls vlc_path2uri itself); vlc.input.add_subtitle_mrl(uri, autoselect)
   takes an MRL. Both go through input_AddSlave(), i.e. asynchronously: the
   new spu ES shows up in the "spu-es" list a little later.
 * vlc.strings.make_path(uri) (vlc_uri2path) and decode_uri/make_uri exist.
 * vlc.osd.message(text, channel, position, duration_us); the channel comes
   from vlc.osd.channel_register(). Silently does nothing without a vout.
 * vlc.config.userdatadir() exists; vlc.io.mkdir(path, "0700") is not
   recursive and returns (status, errno).
 * os.rename/os.remove/os.execute/os.getenv are the standard ones (on
   Windows os.rename fails if the target exists -> remove + retry).
--]==========================================================================]

local M = {}

M.VERSION = "0.1.0"
M.TICK_US = 500000            -- main loop period
M.DEBOUNCE_US = 1500000       -- wait this long after a track change
M.ADD_TIMEOUT_US = 10000000   -- wait at most this long for an added track to appear
M.OSD_PROGRESS_US = 3000000   -- min interval between progress OSD updates
M.HEARTBEAT_CHECK_US = 2000000
M.STATE_REFRESH_S = 5         -- rewrite intf_state at least this often
M.HEARTBEAT_MAX_AGE = 10      -- daemon considered alive if heartbeat age <= this
M.DAEMON_GRACE_US = 3000000   -- time a job may wait before we complain about the daemon
M.OSD_DURATION = 3000000

M.execute = os.execute        -- replaceable in tests
M.getenv = os.getenv

local S                       -- runtime state (see M.reset)

---------------------------------------------------------------- logging

local function log_dbg(msg)
    pcall(vlc.msg.dbg, "[subsync] " .. tostring(msg))
end
local function log_info(msg)
    pcall(vlc.msg.info, "[subsync] " .. tostring(msg))
end
local function log_err(msg)
    pcall(vlc.msg.err, "[subsync] " .. tostring(msg))
end
M.log_dbg, M.log_info, M.log_err = log_dbg, log_info, log_err

---------------------------------------------------------------- helpers

local function now_us()
    return vlc.misc.mdate()
end

local function is_windows()
    return package.config:sub(1, 1) == "\\"
end

local function one_line(s)
    s = tostring(s or "")
    return (s:gsub("[\1-\31\127]", " "))
end

local function trim(s)
    return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function join(dir, name)
    local sep = is_windows() and "\\" or "/"
    if dir:sub(-1) == "/" or dir:sub(-1) == "\\" then
        return dir .. name
    end
    return dir .. sep .. name
end
M.join = join

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end
M.file_exists = file_exists

function M.parse_kv(data)
    local t = {}
    if not data then return t end
    data = data:gsub("^\239\187\191", "") -- UTF-8 BOM
    for line in (data .. "\n"):gmatch("([^\n]*)\n") do
        line = line:gsub("\r$", "")
        if not line:match("^%s*[#;]") then
            local k, v = line:match("^%s*([%w_%.%-]+)%s*=(.*)$")
            if k then t[k] = trim(v) end
        end
    end
    return t
end

function M.read_kv(path)
    local f = io.open(path, "rb")
    if not f then return nil end
    local data = f:read("*a")
    f:close()
    return M.parse_kv(data)
end

-- Write key=value lines atomically: <path>.tmp, then rename over <path>.
-- `kv` is an array of {key, value} pairs (keeps the order stable).
function M.write_kv(path, kv)
    local lines = {}
    for _, pair in ipairs(kv) do
        lines[#lines + 1] = pair[1] .. "=" .. one_line(pair[2])
    end
    local tmp = path .. ".tmp"
    local f, err = io.open(tmp, "wb")
    if not f then
        return false, "cannot write " .. tmp .. ": " .. tostring(err)
    end
    f:write(table.concat(lines, "\n") .. "\n")
    f:close()
    local ok, rerr = os.rename(tmp, path)
    if not ok then
        os.remove(path) -- Windows: rename does not replace
        ok, rerr = os.rename(tmp, path)
    end
    if not ok then
        os.remove(tmp)
        return false, "cannot rename " .. tmp .. ": " .. tostring(rerr)
    end
    return true
end

local function mkdir(path)
    if vlc.io and type(vlc.io.mkdir) == "function" then
        pcall(vlc.io.mkdir, path, "0700")
    end
end

local function percent_decode(s)
    return (s:gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end))
end
M.percent_decode = percent_decode

-- file:// URI -> local path, or nil for anything that is not a local file.
function M.uri_to_path(uri)
    if type(uri) ~= "string" or not uri:lower():match("^file://") then
        return nil
    end
    if vlc.strings and type(vlc.strings.make_path) == "function" then
        local ok, p = pcall(vlc.strings.make_path, uri)
        if ok and type(p) == "string" and p ~= "" then
            return p
        end
    end
    local rest = uri:sub(8)
    local host, path = rest:match("^([^/]*)(/.*)$")
    if not path then return nil end
    path = percent_decode(path)
    if host ~= "" and host:lower() ~= "localhost" then
        path = "//" .. host .. path          -- UNC share
    end
    if is_windows() then
        if path:match("^/%a:") then path = path:sub(2) end
        path = path:gsub("/", "\\")
    end
    return path
end

---------------------------------------------------------------- environment

function M.queue_dir()
    local base = vlc.config.userdatadir()
    return join(base, "subsync")
end

function M.is_sandboxed()
    local snap = M.getenv("SNAP")
    local flat = M.getenv("FLATPAK_ID")
    if (snap and snap ~= "") or (flat and flat ~= "") then return true end
    local q = S and S.q or ""
    if q:find("/snap/", 1, true) or q:find("/.var/app/", 1, true) then
        return true
    end
    return false
end

function M.helper_candidates()
    local home = M.getenv("HOME") or ""
    local list = {}
    if home ~= "" then
        list[#list + 1] = home .. "/.local/bin/vlc-subsync"
        list[#list + 1] = home .. "/.cargo/bin/vlc-subsync"
    end
    list[#list + 1] = "/usr/local/bin/vlc-subsync"
    list[#list + 1] = "/opt/homebrew/bin/vlc-subsync"
    list[#list + 1] = "/usr/bin/vlc-subsync"
    return list
end

local function shell_quote(s)
    return "'" .. s:gsub("'", "'\\''") .. "'"
end

-- Try (once) to start the daemon ourselves. Only on non-Windows, non-sandboxed
-- VLC: os.execute flashes a console on Windows and is denied in snap/flatpak.
function M.try_autostart()
    if S.autostart_tried then return false end
    S.autostart_tried = true
    if is_windows() or M.is_sandboxed() then
        log_dbg("not trying to start the helper (windows or sandboxed VLC)")
        return false
    end
    for _, p in ipairs(M.helper_candidates()) do
        if file_exists(p) then
            local cmd = shell_quote(p) .. " serve >/dev/null 2>&1 &"
            log_info("starting helper: " .. cmd)
            pcall(M.execute, cmd)
            return true
        end
    end
    log_dbg("vlc-subsync not found in the usual locations")
    return false
end

function M.daemon_alive(force)
    local t = now_us()
    if force or not S.hb_checked or t - S.hb_checked >= M.HEARTBEAT_CHECK_US then
        S.hb_checked = t
        local hb = M.read_kv(join(S.q, "heartbeat"))
        local ht = hb and tonumber(hb.time)
        S.hb_alive = (ht ~= nil) and (os.time() - ht <= M.HEARTBEAT_MAX_AGE)
        S.hb_version = hb and hb.version or nil
    end
    return S.hb_alive
end

---------------------------------------------------------------- OSD

function M.osd(text)
    log_dbg("OSD: " .. text)
    if not (vlc.osd and type(vlc.osd.message) == "function") then return end
    if not S.osd_channel and type(vlc.osd.channel_register) == "function" then
        local ok, ch = pcall(vlc.osd.channel_register)
        if ok and type(ch) == "number" then S.osd_channel = ch end
    end
    pcall(vlc.osd.message, text, S.osd_channel or 1, "top-right", M.OSD_DURATION)
end

---------------------------------------------------------------- intf_state

function M.set_state(state, message, last_result)
    if state ~= S.state or message ~= S.message
        or (last_result ~= nil and last_result ~= S.last_result) then
        S.state = state
        S.message = message or ""
        if last_result ~= nil then S.last_result = last_result end
        S.state_dirty = true
    end
end

function M.write_state(force)
    local t = os.time()
    if not (force or S.state_dirty or not S.state_written
            or t - S.state_written >= M.STATE_REFRESH_S) then
        return
    end
    local ok, err = M.write_kv(join(S.q, "intf_state"), {
        { "time", t },
        { "version", M.VERSION },
        { "state", S.state },
        { "message", S.message },
        { "last_result", S.last_result or "" },
        { "auto", S.auto and 1 or 0 },
        { "media", S.media_path or "" },
        { "job", S.job and S.job.id or "" },
        { "daemon", S.hb_alive and 1 or 0 },
        -- sync_now_mode values this intf understands (older intfs lack the key)
        { "modes", "fast,thorough,exhaustive" },
    })
    if ok then
        S.state_written = t
        S.state_dirty = false
    elseif not S.state_err_logged then
        S.state_err_logged = true
        log_err(err)
    end
end

---------------------------------------------------------------- control

M.MODES = { fast = true, thorough = true, exhaustive = true }

-- `m` if it is a known sync mode, else "" (= the daemon's configured mode).
function M.valid_mode(m)
    m = trim(tostring(m or "")):lower()
    if M.MODES[m] then return m end
    return ""
end

-- OSD text of a starting job; slow modes warn that they take a while.
function M.syncing_text(mode)
    if mode == "exhaustive" or mode == "thorough" then
        return "Syncing subtitles (" .. mode .. ", may take a while)…"
    end
    return "Syncing subtitles…"
end

-- Returns true (plus the requested mode, "" = default) when "Sync now" was asked
-- for since the last call, false otherwise. The extension writes
-- sync_now=<counter> and, for "Sync now (exhaustive)", sync_now_mode=exhaustive.
function M.read_control()
    local c = M.read_kv(join(S.q, "control"))
    if not c then
        if S.sync_now_seen == nil then S.sync_now_seen = 0 end
        return false
    end
    if c.auto ~= nil and c.auto ~= "" then
        local auto = not (c.auto == "0" or c.auto:lower() == "off" or c.auto:lower() == "false")
        if auto ~= S.auto then
            S.auto = auto
            S.state_dirty = true
            log_info("auto-sync " .. (auto and "enabled" or "disabled"))
            if not auto then S.pending = nil end
        end
    end
    local n = tonumber(c.sync_now) or 0
    if S.sync_now_seen == nil then
        S.sync_now_seen = n -- baseline: ignore requests made before we started
        return false
    end
    if n > S.sync_now_seen then
        S.sync_now_seen = n
        return true, M.valid_mode(c.sync_now_mode)
    end
    if n < S.sync_now_seen then S.sync_now_seen = n end -- counter reset
    return false
end

---------------------------------------------------------------- tracks

-- Returns ids, labels arrays (may be empty) for an ES variable.
local function es_list(input, var)
    local ok, ids, labels = pcall(vlc.var.get_list, input, var)
    if not ok or type(ids) ~= "table" then return {}, {} end
    if type(labels) ~= "table" then labels = {} end
    return ids, labels
end

local function var_get(input, var)
    local ok, v = pcall(vlc.var.get, input, var)
    if ok then return tonumber(v) end
    return nil
end

-- Snapshot of the current track situation of `input`.
function M.snapshot(input)
    local snap = {}
    snap.audio_ids, snap.audio_labels = es_list(input, "audio-es")
    snap.spu_ids, snap.spu_labels = es_list(input, "spu-es")
    snap.audio = var_get(input, "audio-es") or -1
    snap.spu = var_get(input, "spu-es") or -1
    snap.spu_set = {}
    for _, id in ipairs(snap.spu_ids) do snap.spu_set[id] = true end
    return snap
end

-- Ordinal (0-based) of `id` in `ids`, skipping -1 and ids in `skip`.
function M.ordinal(ids, labels, id, skip)
    local n = 0
    for i, v in ipairs(ids) do
        if v ~= -1 and not (skip and skip[v]) then
            if v == id then return n, labels[i] or "" end
            n = n + 1
        end
    end
    return nil
end

-- Resolve the current selection into {audio=ordinal, sub=ordinal, ...} or nil.
function M.selection(snap)
    local sel = { audio_id = snap.audio, spu_id = snap.spu }
    if snap.audio ~= -1 then
        sel.audio, sel.audio_label = M.ordinal(snap.audio_ids, snap.audio_labels, snap.audio)
    end
    if snap.spu ~= -1 then
        local ours = S.ours[snap.spu]
        if ours then
            sel.is_ours = true
            sel.sub, sel.sub_label = ours.sub, ours.sub_label
            sel.source_audio = ours.audio
        else
            sel.sub, sel.sub_label = M.ordinal(snap.spu_ids, snap.spu_labels, snap.spu, S.ours)
        end
    end
    if sel.audio and sel.sub then
        sel.key = sel.audio .. "|" .. sel.sub
    end
    return sel
end

-- ES id of the original (non-ours) subtitle track with ordinal `sub`.
function M.spu_id_for_ordinal(snap, sub)
    local n = 0
    for _, v in ipairs(snap.spu_ids) do
        if v ~= -1 and not S.ours[v] then
            if n == sub then return v end
            n = n + 1
        end
    end
    return nil
end

local function select_spu(input, id)
    local ok, err = pcall(vlc.var.set, input, "spu-es", id)
    if not ok then log_err("cannot select track " .. tostring(id) .. ": " .. tostring(err)) end
    return ok
end

---------------------------------------------------------------- input tracking

function M.reset_input()
    if S.job and S.job.req_path then os.remove(S.job.req_path) end
    S.input_uri = nil
    S.media_path = nil
    S.ours = {}
    S.last_audio, S.last_spu = nil, nil
    S.pending = nil
    S.job = nil
    S.adding = nil
    S.non_file_logged = nil
end

local function memo_for(uri)
    local m = S.memo[uri]
    if not m then
        m = {}
        S.memo[uri] = m
    end
    return m
end

---------------------------------------------------------------- requests

function M.new_id()
    S.counter = S.counter + 1
    return tostring(os.time()) .. "_" .. tostring(S.counter)
end

function M.ensure_dirs()
    mkdir(S.q)
    mkdir(join(S.q, "requests"))
    mkdir(join(S.q, "jobs"))
    mkdir(join(S.q, "out"))
end

function M.submit(sel, force, mode)
    mode = M.valid_mode(mode)
    -- one job in flight per input: a newer request supersedes the old one
    if S.job then
        if S.job.req_path then os.remove(S.job.req_path) end
        log_dbg("superseding job " .. S.job.id)
        S.job = nil
    end
    M.ensure_dirs()
    local id = M.new_id()
    local path = join(join(S.q, "requests"), id .. ".req")
    local kv = {
        { "version", 1 },
        { "id", id },
        { "media", S.media_path },
        { "audio_index", sel.audio },
        { "audio_label", sel.audio_label or "" },
        { "sub_index", sel.sub },
        { "sub_label", sel.sub_label or "" },
        { "sub_path", "" },
        { "force", force and 1 or 0 },
    }
    if mode ~= "" then kv[#kv + 1] = { "mode", mode } end
    local ok, err = M.write_kv(path, kv)
    if not ok then
        log_err(err)
        M.osd("SubSync: cannot write request (see log)")
        M.set_state("error", "cannot write request", nil)
        return nil
    end
    log_info(string.format("request %s: audio=%d sub=%d mode=%s media=%s",
        id, sel.audio, sel.sub, mode ~= "" and mode or "default", S.media_path))
    S.job = {
        id = id, key = sel.key, audio = sel.audio, sub = sel.sub,
        sub_label = sel.sub_label, req_path = path, started = now_us(), mode = mode,
        status_path = join(join(S.q, "jobs"), id .. ".status"),
    }
    local text = M.syncing_text(mode)
    M.osd(text)
    S.last_progress_osd = now_us()
    M.set_state("syncing", text, nil)
    if not M.daemon_alive(true) then
        M.warn_daemon()
    end
    return S.job
end

function M.warn_daemon()
    if S.daemon_warned then return end
    S.daemon_warned = true
    local started = M.try_autostart()
    if started then
        M.osd("SubSync helper not running – starting it…")
    else
        M.osd("SubSync helper not running")
    end
    log_err("helper daemon not running (no fresh heartbeat in " .. S.q .. ")")
end

-- Load a synced subtitle file into the current input; selection of the new ES
-- happens once it shows up (see M.check_adding).
function M.load_output(_input, snap, entry, select)
    local output = entry.output
    local ok, err = pcall(vlc.input.add_subtitle, output, select)
    if not ok then
        log_dbg("add_subtitle(path) failed: " .. tostring(err))
        local uri = output
        if vlc.strings and type(vlc.strings.make_uri) == "function" then
            local ok2, u = pcall(vlc.strings.make_uri, output)
            if ok2 and u then uri = u end
        end
        local f = vlc.input.add_subtitle_mrl or vlc.input.add_subtitle
        ok, err = pcall(f, uri, select)
    end
    if not ok then
        log_err("cannot add subtitle " .. output .. ": " .. tostring(err))
        M.osd("SubSync: could not load synced subtitles")
        M.set_state("error", "could not load synced subtitles", nil)
        return false
    end
    local before = {}
    for _, id in ipairs(snap.spu_ids) do before[id] = true end
    S.adding = {
        before = before, entry = entry, select = select,
        deadline = now_us() + M.ADD_TIMEOUT_US,
    }
    log_dbg("added subtitle " .. output .. ", waiting for its track")
    return true
end

-- Called every tick while an added subtitle is pending; returns true while
-- still waiting (triggers are suspended meanwhile).
function M.check_adding(input, snap)
    local a = S.adding
    local new_ids = {}
    for _, id in ipairs(snap.spu_ids) do
        if id ~= -1 and not a.before[id] and not S.ours[id] then
            new_ids[#new_ids + 1] = id
        end
    end
    if #new_ids == 0 then
        if now_us() > a.deadline then
            log_err("synced subtitle track did not appear")
            M.set_state("error", "synced track did not appear", nil)
            S.adding = nil
            return false
        end
        return true
    end
    local e = a.entry
    for _, id in ipairs(new_ids) do
        S.ours[id] = { audio = e.audio, sub = e.sub, sub_label = e.sub_label, output = e.output }
    end
    local es = new_ids[1]
    e.es = es
    S.adding = nil
    if a.select and snap.spu ~= es then
        select_spu(input, es)
        snap.spu = es
    end
    -- the new selection must not look like a user change
    S.last_spu, S.last_audio = snap.spu, snap.audio
    if a.select then
        M.osd("Subtitles synced: " .. (e.message ~= "" and e.message or "done"))
    end
    log_info("synced track es=" .. tostring(es) .. " for " .. e.audio .. "|" .. e.sub)
    return false
end

-- Handle a sync trigger for selection `sel`.
function M.fire(input, snap, sel, force)
    local memo = memo_for(S.input_uri)
    local m = memo[sel.key]
    if m and not force then
        if m.applied == false then
            log_dbg("previous sync for " .. sel.key .. " was not applied; not retrying")
            return
        end
        if m.es and snap.spu_set[m.es] and S.ours[m.es] then
            if snap.spu ~= m.es then
                log_dbg("re-selecting synced track " .. m.es)
                select_spu(input, m.es)
                S.last_spu = m.es
                M.osd("Subtitles synced: " .. (m.message ~= "" and m.message or "done"))
            end
            return
        end
        if m.output and file_exists(m.output) then
            M.load_output(input, snap, m, true)
            return
        end
    end
    M.submit(sel, force)
end

function M.handle_status(input, snap)
    local job = S.job
    local st = M.read_kv(job.status_path)
    local t = now_us()
    if not st then
        if t - job.started > M.DAEMON_GRACE_US and not M.daemon_alive() then
            M.warn_daemon()
        end
        return
    end
    job.req_path = nil -- picked up by the daemon
    local state = st.state or "queued"
    if state == "queued" or state == "running" then
        local p = tonumber(st.progress) or 0
        local msg = st.message or ""
        local head = M.syncing_text(job.mode):gsub("…$", "")
        local text = string.format("%s… %d%%", head, math.floor(p * 100 + 0.5))
        if msg ~= "" then text = text .. " – " .. msg end
        M.set_state("syncing", text, nil)
        if text ~= job.last_text and t - (S.last_progress_osd or 0) >= M.OSD_PROGRESS_US then
            job.last_text = text
            S.last_progress_osd = t
            M.osd(text)
        end
        return
    end
    S.job = nil
    local memo = memo_for(S.input_uri)
    if state == "done" then
        local message = st.message or ""
        if st.applied == "1" and st.output and st.output ~= "" then
            local entry = {
                audio = job.audio, sub = job.sub, sub_label = job.sub_label,
                output = st.output, message = message, applied = true,
            }
            memo[job.key] = entry
            -- only auto-select if the user still has that combination selected
            local sel = M.selection(snap)
            local select = (sel.key == job.key)
            M.set_state("done", "Subtitles synced", "synced: " .. message)
            M.load_output(input, snap, entry, select)
        else
            memo[job.key] = { applied = false, message = message }
            local text = "SubSync: not synced"
            if message ~= "" then text = text .. " (" .. message .. ")" end
            M.osd(text .. " – keeping original timing")
            M.set_state("done", "not applied", "not applied: " .. message)
        end
    elseif state == "error" then
        local message = st.message or "unknown error"
        M.osd("SubSync error: " .. message)
        M.set_state("error", message, "error: " .. message)
        log_err("job " .. job.id .. " failed: " .. message)
    else
        log_dbg("unknown job state " .. tostring(state))
    end
end

---------------------------------------------------------------- tick

function M.tick()
    local t = now_us()
    local sync_now, sync_mode = M.read_control()

    local get_input = vlc.object and vlc.object.input
    if type(get_input) ~= "function" then
        if not S.no_input_api then
            S.no_input_api = true
            log_err("vlc.object.input() not available in this VLC version; idling")
            M.set_state("error", "unsupported VLC version", nil)
        end
        M.write_state()
        return
    end
    local input = get_input()
    if not input then
        if S.input_uri then M.reset_input() end
        M.set_state(S.auto and "idle" or "disabled", "no input", nil)
        M.write_state()
        return
    end

    local item = vlc.input.item()
    local uri = item and item:uri() or nil
    if uri ~= S.input_uri then
        M.reset_input()
        S.input_uri = uri
        S.media_path = M.uri_to_path(uri)
        if S.media_path then log_dbg("new input: " .. S.media_path) end
    end
    if not S.media_path then
        if not S.non_file_logged then
            S.non_file_logged = true
            log_dbg("not a local file, ignoring: " .. tostring(uri))
        end
        M.set_state("idle", "not a local file", nil)
        M.write_state()
        return
    end

    local snap = M.snapshot(input)

    -- forget our tracks that vanished (input restarted / re-opened)
    for id, o in pairs(S.ours) do
        if not snap.spu_set[id] then
            S.ours[id] = nil
            local m = S.memo[S.input_uri]
            if m then
                for _, e in pairs(m) do
                    if e.es == id then e.es = nil end
                end
            end
            log_dbg("synced track " .. id .. " (" .. tostring(o.output) .. ") gone")
        end
    end

    if S.adding and M.check_adding(input, snap) then
        M.write_state()
        return
    end

    if S.job then M.handle_status(input, snap) end
    if S.adding then
        M.write_state()
        return
    end

    local sel = M.selection(snap)
    local audio_changed = snap.audio ~= S.last_audio
    local spu_changed = snap.spu ~= S.last_spu
    S.last_audio, S.last_spu = snap.audio, snap.spu
    if audio_changed or spu_changed then
        log_dbg(string.format("tracks: audio-es=%s (#%s) spu-es=%s (#%s)%s",
            tostring(snap.audio), tostring(sel.audio), tostring(snap.spu), tostring(sel.sub),
            sel.is_ours and " [synced]" or ""))
    end

    if (audio_changed or spu_changed) and S.auto then
        if not sel.key then
            S.pending = nil -- subtitles disabled (or no audio): nothing to do
        elseif sel.is_ours and not audio_changed then
            S.pending = nil -- the user (or we) selected a synced track
        else
            S.pending = { due = t + M.DEBOUNCE_US, key = sel.key, force = false }
            log_dbg("change detected (" .. sel.key .. "), debouncing")
        end
    end

    if sync_now then
        if sel.key then
            -- re-syncing an already synced track: bypass the daemon's cache
            S.pending = { due = t, key = sel.key, force = sel.is_ours and true or false,
                          explicit = true, mode = sync_mode }
        else
            M.osd("SubSync: select a subtitle track first")
        end
    end

    if S.pending and t >= S.pending.due then
        local p = S.pending
        S.pending = nil
        if sel.key == p.key then
            local fsel = sel
            if sel.is_ours then
                -- sync from the original source track, not from our output
                fsel = {
                    audio = sel.audio, audio_label = sel.audio_label,
                    sub = sel.sub, sub_label = sel.sub_label, key = sel.key,
                }
            end
            if p.explicit then
                M.submit(fsel, p.force, p.mode)
            else
                M.fire(input, snap, fsel, p.force)
            end
        else
            log_dbg("selection changed during debounce; dropping trigger")
        end
    end

    -- "done"/"error" stay visible until the next job starts
    if not S.job and not S.adding and S.state ~= "error" and S.state ~= "done" then
        M.set_state(S.auto and "idle" or "disabled", "", nil)
    end
    M.write_state()
end

---------------------------------------------------------------- lifecycle

function M.reset()
    S = {
        counter = 0, auto = true, ours = {}, memo = {},
        state = "idle", message = "", last_result = "",
    }
    S.q = M.queue_dir()
    M.state = S
    return S
end

function M.init()
    M.reset()
    M.ensure_dirs()
    log_info("started, queue dir " .. S.q)
    M.write_state(true)
end

function M.safe_tick()
    local ok, err = pcall(M.tick)
    if not ok then
        S.err_count = (S.err_count or 0) + 1
        if S.err_count <= 20 or S.err_count % 100 == 0 then
            log_err("tick failed: " .. tostring(err))
        end
    end
    return ok
end

function M.run()
    local ok, err = pcall(M.init)
    if not ok then
        log_err("init failed: " .. tostring(err))
        if not S then return end
    end
    while true do
        if type(vlc.misc.should_die) == "function" then
            local okd, die = pcall(vlc.misc.should_die)
            if okd and die then break end
        end
        M.safe_tick()
        -- VLC 3 interrupts mwait() with an error when the interface closes
        local okw = pcall(vlc.misc.mwait, vlc.misc.mdate() + M.TICK_US)
        if not okw then break end
    end
    pcall(function()
        -- withdraw a request the daemon has not picked up yet
        if S.job and S.job.req_path then os.remove(S.job.req_path) end
        M.set_state("stopped", "", nil)
        M.write_state(true)
    end)
    log_dbg("stopped")
end

if rawget(_G, "SUBSYNC_TEST") then
    _G.subsync = M
else
    M.run()
end
