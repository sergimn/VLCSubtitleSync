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
 * os.execute(cmd) is C system(): `/bin/sh -c cmd` on POSIX, `cmd.exe /c cmd` on
   Windows. From a GUI process like VLC, cmd.exe gets a console window of its
   own, which may flash briefly; `start "" /B` makes it return at once (see
   DESIGN.md "Lifecycle" for why this is the least visible option).
 * Input variables "time" and "spu-delay" are VLC_VAR_INTEGER in microseconds
   (src/input/var.c: var_Create(p_input, "time"/"spu-delay", VLC_VAR_INTEGER);
   input.c: spu-delay = sub-delay (1/10 s) * 100000). "time" is the playback
   position (es_out.c ES_OUT_SET_TIMES subtracts the buffering). A positive
   spu-delay shows subtitles later: decoder.c DecoderFixTs adds it to every
   subtitle timestamp when the subpicture is queued, so a change only affects
   subtitles decoded afterwards. A negative value also enlarges the input's
   pts_delay (input.c UpdatePtsDelay), and the clock never shrinks it again
   (clock.c input_clock_SetJitter): each new minimum stalls playback once by
   the difference. See DESIGN.md "Delay mode".
--]==========================================================================]

local M = {}

M.VERSION = "1.0.2"
M.TICK_US = 500000            -- main loop period
M.DEBOUNCE_US = 1500000       -- wait this long after a track change
-- After a file opens, VLC (and the user) may still be picking tracks: automatic
-- syncs wait this long before transcribing, unless the user changes a track (that
-- ends the wait) or the helper has a synced result cached (used at once).
M.START_HOLD_US = 10000000
M.START_SETTLE_US = 2000000   -- track changes this soon after audio shows up are VLC's
M.ADD_TIMEOUT_US = 10000000   -- wait at most this long for an added track to appear
M.OSD_PROGRESS_US = 3000000   -- min interval between progress OSD updates
M.HEARTBEAT_CHECK_US = 2000000
M.STATE_REFRESH_S = 5         -- rewrite intf_state at least this often
M.HEARTBEAT_MAX_AGE = 10      -- daemon considered alive if heartbeat age <= this
M.DAEMON_GRACE_US = 3000000   -- time a job may wait before we complain about the daemon
M.SPAWN_RETRY_US = 60000000   -- launcher spawn: wait this long before starting it again
M.SPAWN_MAX = 3               -- launcher spawn: at most this many starts per VLC session
M.OSD_DURATION = 3000000
-- delay mode (experimental): see DESIGN.md "Delay mode"
M.DELAY_TOLERANCE_US = 40000  -- re-set spu-delay when the target moved more than this
M.SEEK_US = 2000000           -- "time" jumped this far from wall-clock progress = seek
M.LOOKAHEAD_S = 1.0           -- see M.delay_update
M.SYNC_MODES = { track = true, delay = true }

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

-- <q>/launcher, written by `vlc-subsync setup` (see DESIGN.md "Lifecycle"):
--   mode=service  a service manager (systemd path unit, launchd) starts the helper
--   mode=spawn    we start `exe args` ourselves (Windows; Linux without systemd)
function M.read_launcher()
    return M.read_kv(join(S.q, "launcher"))
end

-- Shell command that starts the helper in the background and returns at once.
function M.spawn_command(exe, args)
    local parts = {}
    for a in tostring(args or ""):gmatch("%S+") do parts[#parts + 1] = a end
    if is_windows() then
        -- cmd.exe: `start` returns immediately; "" is the (empty) window title.
        -- The helper is a GUI-subsystem exe, so it opens no console of its own.
        local cmd = 'start "" /B "' .. exe .. '"'
        for _, a in ipairs(parts) do cmd = cmd .. ' "' .. a .. '"' end
        return cmd
    end
    local cmd = shell_quote(exe)
    for _, a in ipairs(parts) do cmd = cmd .. " " .. shell_quote(a) end
    return cmd .. " >/dev/null 2>&1 &"
end

-- Start the helper as the launcher file says. Returns true if it was started now
-- or recently (and may still be coming up).
function M.spawn_from_launcher(l)
    if l.mode ~= "spawn" then
        if not S.service_logged then
            S.service_logged = true
            log_dbg("helper is started by a service manager (launcher mode=" ..
                tostring(l.mode) .. ")")
        end
        return false
    end
    if not is_windows() and M.is_sandboxed() then
        log_dbg("sandboxed VLC cannot start the helper")
        return false
    end
    local exe = l.exe or ""
    if exe == "" or not file_exists(exe) then
        if not S.launcher_err_logged then
            S.launcher_err_logged = true
            log_err("helper not found: '" .. exe .. "' (re-run `vlc-subsync setup`)")
        end
        return false
    end
    local t = now_us()
    if S.spawned_at and t - S.spawned_at < M.SPAWN_RETRY_US then
        return true -- started a moment ago: give it time to write its heartbeat
    end
    if S.spawn_count >= M.SPAWN_MAX then return false end
    S.spawned_at = t
    S.spawn_count = S.spawn_count + 1
    local cmd = M.spawn_command(exe, l.args)
    log_info("starting helper: " .. cmd)
    pcall(M.execute, cmd)
    return true
end

-- Heartbeat-triggered start (launcher mode=spawn only): called at startup and
-- every HEARTBEAT_CHECK_US; starts the helper when its heartbeat is not fresh.
function M.check_helper()
    local t = now_us()
    if S.helper_checked and t - S.helper_checked < M.HEARTBEAT_CHECK_US then return end
    S.helper_checked = t
    if M.daemon_alive(true) then return end
    local l = M.read_launcher()
    if l and l.mode == "spawn" then M.spawn_from_launcher(l) end
end

-- Try to start the daemon ourselves. With a launcher file, as it says; without one
-- (set up by an earlier version) once, and only on non-Windows, non-sandboxed VLC.
function M.try_autostart()
    local l = M.read_launcher()
    if l then return M.spawn_from_launcher(l) end
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

-- Errors are always shown; everything else (progress, "Subtitles synced", the
-- helper being started) only with "Show all messages" on (messages=all in
-- <q>/control), see DESIGN.md "On-screen messages".
function M.osd(text)
    log_dbg("OSD: " .. text)
    if not (vlc.osd and type(vlc.osd.message) == "function") then return end
    if not S.osd_channel and type(vlc.osd.channel_register) == "function" then
        local ok, ch = pcall(vlc.osd.channel_register)
        if ok and type(ch) == "number" then S.osd_channel = ch end
    end
    pcall(vlc.osd.message, text, S.osd_channel or 1, "top-right", M.OSD_DURATION)
end

function M.osd_info(text)
    if S.messages_all then
        M.osd(text)
    else
        log_dbg("OSD (hidden, messages=errors): " .. text)
    end
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
        -- how results are applied: track (default) or delay (experimental)
        { "sync_mode", M.current_sync_mode() },
        { "sync_modes", "track,delay" },
        { "delay_active", S.delay and 1 or 0 },
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
    local all = trim(tostring(c.messages or "")):lower() == "all"
    if all ~= (S.messages_all or false) then
        S.messages_all = all
        log_info("on-screen messages: " .. (all and "all" or "errors only"))
    end
    local sm = trim(tostring(c.sync_mode or "")):lower()
    if not M.SYNC_MODES[sm] then sm = nil end
    if sm ~= S.sync_mode_ctl then
        local before = M.current_sync_mode()
        S.sync_mode_ctl = sm
        S.state_dirty = true
        if M.current_sync_mode() ~= before then
            log_info("sync mode " .. M.current_sync_mode())
            S.sync_mode_changed = true
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

-- "track" (load the synced file as an extra track, the default) or "delay"
-- (experimental). The extension's toggle (control sync_mode=) wins over the
-- helper's config (sync_mode= in the last done status).
function M.current_sync_mode()
    if not S then return "track" end
    return S.sync_mode_ctl or S.cfg_sync_mode or "track"
end

---------------------------------------------------------------- delay mode
-- The mapping comes from a done status: segments=<n>, seg<i>=<sub_start>,
-- <sub_end>,<scale>,<offset> (empty bound = open) and optionally
-- seg<i>_knots=<t>:<c>;... with audio = scale*sub + offset + c(sub), all in
-- seconds of the subtitle clock (DESIGN.md "Mapping segments").

M.MAX_SEGMENTS = 256
M.MAX_KNOTS = 4096          -- knots of all segments together (more are ignored)
M.BIAS_MIN_US = 1000        -- spu-delay differences below this are not the user's
-- String variable we create on the input object: "<token>|<bias>|<last set>".
-- A new input object (same file played again, repeat, playlist loop) lacks it,
-- and a restarted intf finds its predecessor's correction in it.
M.MARK_VAR = "subsync-delay"

local function num(s)
    local v = tonumber(s)
    if v == nil or v ~= v or v == math.huge or v == -math.huge then return nil end
    return v
end

-- Local refinement c(t): linear between knots, flat outside (binary search).
function M.seg_correction(seg, t)
    local k = seg.knots
    local n = #k
    if n == 0 then return 0 end
    if t <= k[1][1] then return k[1][2] end
    if t >= k[n][1] then return k[n][2] end
    local lo, hi = 1, n -- k[lo][1] < t <= k[hi][1]
    while hi - lo > 1 do
        local mid = math.floor((lo + hi) / 2)
        if k[mid][1] < t then lo = mid else hi = mid end
    end
    local x0, y0, x1, y1 = k[lo][1], k[lo][2], k[hi][1], k[hi][2]
    if x1 <= x0 then return y0 end
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
end

function M.seg_audio(seg, s)
    return seg.scale * s + seg.offset + M.seg_correction(seg, s)
end

function M.parse_segments(st)
    local n = tonumber(st and st.segments)
    if not n or n < 1 or n > M.MAX_SEGMENTS then return nil end
    local segs = {}
    local total = 0
    for i = 0, n - 1 do
        local v = st["seg" .. i]
        if not v then return nil end
        local a, b, c, d = v:match("^%s*([^,]*),([^,]*),([^,]*),([^,]*)%s*$")
        if not a then return nil end
        local seg = { lo = num(a), hi = num(b), scale = num(c), offset = num(d), knots = {} }
        if (trim(a) ~= "" and not seg.lo) or (trim(b) ~= "" and not seg.hi)
            or not seg.scale or not seg.offset or seg.scale <= 0 then
            return nil
        end
        local k = st["seg" .. i .. "_knots"]
        if k and k ~= "" then
            for item in k:gmatch("[^;]+") do
                local kt, kc = item:match("^%s*([^:]+):(.+)$")
                kt, kc = num(kt), num(kc)
                if not kt or not kc then return nil end
                seg.knots[#seg.knots + 1] = { kt, kc }
            end
            if total + #seg.knots > M.MAX_KNOTS then
                log_err("mapping has more than " .. M.MAX_KNOTS .. " knots; ignoring those of segment " .. i)
                seg.knots = {}
            end
            total = total + #seg.knots
            table.sort(seg.knots, function(x, y) return x[1] < y[1] end)
        end
        -- audio-domain span [alo, ahi), computed once (open bounds -> +-huge)
        seg.alo = seg.lo and M.seg_audio(seg, seg.lo) or -math.huge
        seg.ahi = seg.hi and M.seg_audio(seg, seg.hi) or math.huge
        segs[#segs + 1] = seg
    end
    return segs
end

-- Segment index for audio time T (s): the (last) segment whose audio span
-- contains T; in a gap (forward cut) the upcoming one; after the end the
-- nearest preceding one. O(segments), spans are precomputed.
function M.segment_at(segs, T)
    local hit, up, up_lo, prev, prev_hi
    for i, seg in ipairs(segs) do
        local lo, hi = seg.alo, seg.ahi
        if T >= lo and T < hi then hit = i end
        if lo > T and (not up_lo or lo < up_lo) then up, up_lo = i, lo end
        if hi <= T and (not prev_hi or hi >= prev_hi) then prev, prev_hi = i, hi end
    end
    return hit or up or prev or 1
end

-- Delay (s) for audio time T (s): T - s, where s is the subtitle time that
-- maps to T. Also returns the segment index and s.
function M.delay_at(segs, T)
    local i = M.segment_at(segs, T)
    local seg = segs[i]
    local s = (T - seg.offset) / seg.scale
    if #seg.knots > 0 then
        -- c(s) is bounded (+-0.5 s) and slow: a few fixed-point steps converge
        for _ = 1, 4 do
            s = (T - seg.offset - M.seg_correction(seg, s)) / seg.scale
        end
    end
    return T - s, i, s
end

-- Most negative delay over playback times [0, len] (len <= 0: unknown, then
-- only 0 and the segment boundaries are looked at, so an open last segment
-- drifting further negative is underestimated). Delays are linear within a
-- segment (up to the small knot corrections), so the span ends are enough.
function M.min_delay(segs, len)
    local ts = { 0 }
    if len and len > 0 then ts[#ts + 1] = len end
    for _, seg in ipairs(segs) do
        for _, t in ipairs({ seg.alo, seg.ahi - 0.001 }) do
            if t > 0 and t < math.huge and (not len or len <= 0 or t < len) then
                ts[#ts + 1] = t
            end
        end
    end
    local m = math.huge
    for _, t in ipairs(ts) do
        local d = M.delay_at(segs, t)
        if d < m then m = d end
    end
    return m
end

local function round(x)
    return math.floor(x + 0.5)
end

local function set_spu_delay(input, us)
    local ok, err = pcall(vlc.var.set, input, "spu-delay", us)
    if not ok then log_err("cannot set spu-delay: " .. tostring(err)) end
    return ok
end

local function get_int(input, var)
    local ok, v = pcall(vlc.var.get, input, var)
    if ok then return tonumber(v) end
    return nil
end

-- Our mark on the input: token, bias, last set value (nil if absent).
local function read_mark(input)
    local ok, v = pcall(vlc.var.get, input, M.MARK_VAR)
    if not ok or type(v) ~= "string" or v == "" then return nil end
    local tok, b, l = v:match("^([^|]*)|(-?%d+)|(-?%d+)$")
    if not tok then return nil end
    return tok, tonumber(b), tonumber(l)
end

local function write_mark(input, D)
    if D.no_mark then return end
    local value = ""
    if D.last_set then value = D.token .. "|" .. D.bias .. "|" .. D.last_set end
    local ok = pcall(vlc.var.set, input, M.MARK_VAR, value)
    if not ok then D.no_mark = true end
end

-- Fold a change of spu-delay made by someone else (hotkeys G/H, the Track
-- Synchronization dialog) into the user's bias: it stays on top of ours.
local function fold_user_change(input, D)
    if not D.last_set then return end
    local cur = get_int(input, "spu-delay")
    if cur and math.abs(cur - D.last_set) >= M.BIAS_MIN_US then
        D.bias = D.bias + (cur - D.last_set)
        D.last_set = cur
        write_mark(input, D)
        log_info(string.format("delay mode: user adjusted spu-delay, bias now %d us", D.bias))
    end
end

-- True if `input` is still the input object we started on (our mark is there).
local function same_input(input, D)
    if D.no_mark or not D.last_set then return true end
    local tok = read_mark(input)
    return tok == D.token
end

-- Start correcting the selected original track live. `entry` is the memo
-- entry ({segs, message, ...}), `key` its audio|sub key.
function M.delay_start(input, snap, entry, key)
    if S.delay then M.delay_stop(input, "replaced") end
    local cur = round(get_int(input, "spu-delay") or 0)
    local bias = cur
    local _, mbias, mlast = read_mark(input)
    if mbias and mlast and math.abs(cur - mlast) < M.BIAS_MIN_US then
        -- a previous SubSync run (a restarted intf) left its correction here:
        -- the user's own part is the bias it recorded, not the current value
        bias = mbias
        log_info(string.format("delay mode: found an earlier correction (%d us),"
            .. " user bias %d us", cur, bias))
    end
    S.mark_counter = (S.mark_counter or 0) + 1
    local D = {
        key = key, spu_id = snap.spu, audio_id = snap.audio, segs = entry.segs,
        bias = bias, entry = entry,
        token = tostring(os.time()) .. "_" .. S.mark_counter,
    }
    if not pcall(vlc.var.create, input, M.MARK_VAR, "") then D.no_mark = true end
    S.delay = D
    log_info(string.format("delay mode: start (es=%s, %d segment(s), user bias %d us)",
        tostring(snap.spu), #entry.segs, D.bias))
    M.delay_update(input, true)
    local text = "Subtitles synced (live delay, experimental): "
        .. ((entry.message and entry.message ~= "") and entry.message or "done")
    -- every new low of a negative spu-delay pauses playback by the difference
    local dmin = M.min_delay(entry.segs, (get_int(input, "length") or 0) / 1000000)
    if dmin < -0.5 then
        log_info(string.format("delay mode: needs spu-delay down to %.1f s;"
            .. " VLC pauses playback that long in total while it gets there", dmin))
        text = text .. string.format(" – may pause playback up to %.0f s in total", -dmin)
    end
    M.osd_info(text)
    S.state_dirty = true
end

-- Stop, putting the user's own delay (normally 0) back, including a change
-- the user made since our last tick.
function M.delay_stop(input, why)
    local D = S.delay
    if not D then return end
    S.delay = nil
    S.state_dirty = true
    if input and same_input(input, D) then
        fold_user_change(input, D)
        set_spu_delay(input, D.bias)
        D.last_set = nil
        write_mark(input, D) -- cleared
    end
    log_info(string.format("delay mode: stop (%s), spu-delay back to %d us",
        tostring(why), D.bias))
end

-- One tick of delay mode: follow "time", keep the user's manual adjustments.
-- Returns false if the input turned out to be a new one (state was reset).
function M.delay_check_input(input)
    if not S.delay or same_input(input, S.delay) then return true end
    -- same file, new input object (replay, repeat, loop): its spu-delay is
    -- fresh, nothing to restore or fold. Start over as for a new input (the
    -- next tick); the mapping is remembered per media, no new request.
    log_info("delay mode: new input object for the same media, starting over")
    S.delay = nil
    M.reset_input()
    return false
end

function M.delay_update(input, force)
    if not M.delay_check_input(input) then return false end
    local D = S.delay
    local T = get_int(input, "time")
    if not T then return true end
    fold_user_change(input, D)
    local wall = now_us()
    local seek = false
    if D.last_T then
        local jump = (T - D.last_T) - (wall - D.last_wall)
        seek = jump > M.SEEK_US or jump < -M.SEEK_US
    end
    D.last_T, D.last_wall = T, wall
    -- A subtitle takes its delay when it is decoded, ahead of its display:
    -- about the input caching plus any negative delay (VLC buffers that much
    -- more) plus half our update interval. Aim at the subtitles shown then.
    local now_s = T / 1000000
    local d0 = M.delay_at(D.segs, now_s)
    local lead = M.LOOKAHEAD_S + math.max(0, -d0)
    local d, i = M.delay_at(D.segs, now_s + lead)
    local target = round(d * 1000000) + D.bias
    if force or not D.last_set or (seek and target ~= D.last_set)
        or math.abs(target - D.last_set) > M.DELAY_TOLERANCE_US then
        if set_spu_delay(input, target) then
            log_dbg(string.format("spu-delay=%d us (time=%.3fs seg=%d bias=%d us%s)",
                target, T / 1000000, i - 1, D.bias, seek and " seek" or ""))
            D.last_set = target
            write_mark(input, D)
        end
    end
    return true
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
    S.hold_until, S.settle_until, S.cache_missed = nil, nil, {}
    -- the old input (and its spu-delay variable) is gone; a new input starts
    -- from the sub-delay option again, so there is nothing to restore
    if S.delay then log_dbg("delay mode: input gone") end
    S.delay = nil
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

-- `probe`: only ask the helper for a cached result (cache_only=1), see M.request.
function M.submit(sel, force, mode, probe)
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
    if probe then kv[#kv + 1] = { "cache_only", 1 } end
    local ok, err = M.write_kv(path, kv)
    if not ok then
        log_err(err)
        M.osd("SubSync: cannot write request (see log)")
        M.set_state("error", "cannot write request", nil)
        return nil
    end
    log_info(string.format("request %s: audio=%d sub=%d mode=%s%s media=%s",
        id, sel.audio, sel.sub, mode ~= "" and mode or "default",
        probe and " (cache only)" or "", S.media_path))
    S.job = {
        id = id, key = sel.key, audio = sel.audio, sub = sel.sub,
        audio_label = sel.audio_label, sub_label = sel.sub_label, force = force,
        req_path = path, started = now_us(), mode = mode, probe = probe,
        status_path = join(join(S.q, "jobs"), id .. ".status"),
    }
    if probe then
        -- quiet: nothing is transcribed, and a hit shows "Subtitles synced"
        M.set_state("waiting", "checking for a cached result", nil)
    else
        local text = M.syncing_text(mode)
        M.osd_info(text)
        S.last_progress_osd = now_us()
        M.set_state("syncing", text, nil)
    end
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
        M.osd_info("SubSync helper not running – starting it…")
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
        M.osd_info("Subtitles synced: " .. (e.message ~= "" and e.message or "done"))
    end
    log_info("synced track es=" .. tostring(es) .. " for " .. e.audio .. "|" .. e.sub)
    return false
end

-- True while automatic syncs of the current input wait after it opened.
function M.holding(t)
    return S.hold_until ~= nil and t < S.hold_until
end

-- An automatic (unforced) sync request for `sel`. While holding, only ask the helper
-- for a cached result (once per tracks); on a miss the trigger waits for the hold to
-- end (M.handle_status), then this sends the real request.
function M.request(sel)
    local t = now_us()
    if not M.holding(t) then
        M.submit(sel, false)
    elseif S.cache_missed[sel.key] then
        M.wait_hold(sel.key)
    else
        M.submit(sel, false, nil, true)
    end
end

function M.wait_hold(key)
    S.pending = { due = S.hold_until, key = key, force = false, hold = true }
    log_dbg(string.format("waiting %.1f s before syncing %s (file just opened)",
        (S.hold_until - now_us()) / 1000000, key))
    M.set_state("waiting", "waiting a few seconds after opening the file", nil)
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
        if M.current_sync_mode() == "delay" then
            if m.segs and not S.ours[snap.spu] then
                if not (S.delay and S.delay.key == sel.key and S.delay.spu_id == snap.spu) then
                    M.delay_start(input, snap, m, sel.key)
                end
                return
            end
            -- no mapping remembered (synced in track mode by an older helper):
            -- ask the helper again; it re-syncs cached results without one
            M.request(sel)
            return
        end
        if m.es and snap.spu_set[m.es] and S.ours[m.es] then
            if snap.spu ~= m.es then
                log_dbg("re-selecting synced track " .. m.es)
                select_spu(input, m.es)
                S.last_spu = m.es
                M.osd_info("Subtitles synced: " .. (m.message ~= "" and m.message or "done"))
            end
            return
        end
        if m.output and file_exists(m.output) then
            M.load_output(input, snap, m, true)
            return
        end
    end
    if force then
        M.submit(sel, true)
    else
        M.request(sel)
    end
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
    if job.probe and (state == "miss" or state == "error") then
        -- nothing cached (an error is reported again by the real request)
        S.job = nil
        S.cache_missed[job.key] = true
        log_dbg("no cached result for " .. job.key .. " (" .. tostring(st.message) .. ")")
        if not S.pending and M.selection(snap).key == job.key then
            if M.holding(t) then
                M.wait_hold(job.key)
            else
                S.pending = { due = t, key = job.key, force = false }
            end
        end
        return
    end
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
            M.osd_info(text)
        end
        return
    end
    S.job = nil
    local memo = memo_for(S.input_uri)
    if state == "done" then
        local message = st.message or ""
        if st.sync_mode and M.SYNC_MODES[st.sync_mode] and st.sync_mode ~= S.cfg_sync_mode then
            S.cfg_sync_mode = st.sync_mode -- the helper's configured mode
            S.state_dirty = true
        end
        local segs = M.parse_segments(st)
        if st.applied == "1" and M.current_sync_mode() == "delay" then
            -- experimental: no extra track, correct the original one live
            if not segs then
                -- nothing is remembered for this key, so a later selection asks again
                memo[job.key] = nil
                if not job.force then
                    -- e.g. a result cached before mappings were sent: ask for a
                    -- fresh sync once (bypasses the helper's cache)
                    log_info("job " .. job.id .. ": result has no mapping; re-syncing")
                    M.submit({
                        audio = job.audio, audio_label = job.audio_label, sub = job.sub,
                        sub_label = job.sub_label, key = job.key,
                    }, true, job.mode)
                    return
                end
                M.osd("SubSync: live delay needs a newer helper – keeping original timing")
                M.set_state("error", "no mapping in result (helper too old?)", nil)
                log_err("job " .. job.id .. ": delay mode but the status has no segments")
                return
            end
            local entry = {
                audio = job.audio, sub = job.sub, sub_label = job.sub_label,
                output = st.output, message = message, applied = true, segs = segs,
            }
            memo[job.key] = entry
            M.set_state("done", "Subtitles synced (live delay)", "synced (live delay): " .. message)
            local sel = M.selection(snap)
            if sel.key == job.key and not sel.is_ours then
                M.delay_start(input, snap, entry, job.key)
            end
        elseif st.applied == "1" and st.output and st.output ~= "" then
            local entry = {
                audio = job.audio, sub = job.sub, sub_label = job.sub_label,
                output = st.output, message = message, applied = true, segs = segs,
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

-- The sync mode was toggled while playing: move the current result over.
function M.switch_sync_mode(input, snap, sel)
    if M.current_sync_mode() == "track" then
        local D = S.delay
        if not D then return end
        M.delay_stop(input, "sync mode track")
        local e = D.entry
        if e.es and snap.spu_set[e.es] and S.ours[e.es] then
            -- the synced track is still loaded: go back to it, do not add it again
            if snap.spu ~= e.es then
                select_spu(input, e.es)
                snap.spu = e.es
            end
            S.last_spu = e.es
        elseif e.output and file_exists(e.output) then
            M.load_output(input, snap, e, true)
        end
        return
    end
    -- delay: leave our synced track for the original one, corrected live
    if not (sel.is_ours and sel.key) then return end
    local m = S.memo[S.input_uri] and S.memo[S.input_uri][sel.key]
    if not (m and m.segs) then return end
    local orig = M.spu_id_for_ordinal(snap, sel.sub)
    if not orig then return end
    select_spu(input, orig)
    snap.spu = orig
    S.last_spu = orig
    M.delay_start(input, snap, m, sel.key)
end

---------------------------------------------------------------- tick

function M.tick()
    local t = now_us()
    M.check_helper()
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
        if M.START_HOLD_US > 0 then S.hold_until = t + M.START_HOLD_US end
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

    -- delay mode: a replay of the same file is a new input object
    if not M.delay_check_input(input) then
        M.write_state()
        return
    end
    -- delay mode: the corrected combination is no longer selected -> give the
    -- user's own delay back (re-selecting it re-applies the remembered mapping)
    if S.delay and (snap.spu ~= S.delay.spu_id or snap.audio ~= S.delay.audio_id) then
        M.delay_stop(input, "track switch")
    end
    if S.sync_mode_changed then
        S.sync_mode_changed = nil
        M.switch_sync_mode(input, snap, sel)
    end
    if S.delay and not M.delay_update(input, false) then
        M.write_state()
        return
    end

    if S.hold_until then
        if t >= S.hold_until then
            S.hold_until = nil
        elseif not S.settle_until then
            -- VLC picks its default tracks as the streams show up
            if snap.audio ~= -1 then S.settle_until = t + M.START_SETTLE_US end
        elseif (audio_changed or spu_changed) and t >= S.settle_until then
            -- the user picked a track: sync it without waiting
            S.hold_until = nil
            log_dbg("track changed by the user; not waiting any longer")
        end
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
        S.hold_until = nil
        if sel.key then
            -- re-syncing an already synced track: bypass the daemon's cache
            local resync = sel.is_ours or (S.delay ~= nil and S.delay.key == sel.key)
            S.pending = { due = t, key = sel.key, force = resync and true or false,
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
    if not S.job and not S.adding and S.state ~= "error" and S.state ~= "done"
        and not (S.pending and S.pending.hold) then
        M.set_state(S.auto and "idle" or "disabled", "", nil)
    end
    M.write_state()
end

---------------------------------------------------------------- lifecycle

function M.reset()
    S = {
        counter = 0, auto = true, ours = {}, memo = {}, spawn_count = 0, cache_missed = {},
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
    M.check_helper()
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
    M.shutdown()
end

-- Interface closing: give the input its user's delay back (VLC keeps the
-- variable until the input ends), withdraw a request not picked up yet, and
-- record state=stopped in intf_state.
function M.shutdown()
    pcall(function()
        if S.delay then
            local get_input = vlc.object and vlc.object.input
            local input = type(get_input) == "function" and get_input() or nil
            M.delay_stop(input, "exit")
        end
    end)
    pcall(function()
        if S.job and S.job.req_path then os.remove(S.job.req_path) end
        M.set_state("stopped", "", nil)
        M.write_state(true)
    end)
    log_dbg("stopped")
end

-- plain globals only, as in the extension (VLC runs extensions without rawget)
if SUBSYNC_TEST then
    subsync = M
else
    M.run()
end
