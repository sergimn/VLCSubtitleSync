-- Minimal mock of the VLC 3.0 Lua API used by the SubSync scripts.
-- Usage (from the Python harness):  local mock = dofile("vlc_mock.lua").new{...}
--                                   vlc = mock.vlc
-- The mock keeps all state in the returned table so tests can inspect/change it.

local Mock = {}

local function copy_tracks(list)
    local ids, labels = { -1 }, { "Disable" }
    for _, t in ipairs(list) do
        ids[#ids + 1] = t.id
        labels[#labels + 1] = t.label
    end
    return ids, labels
end

function Mock.new(opts)
    opts = opts or {}
    local m = {
        now = 1000000,          -- fake clock, microseconds
        waits = 0,
        interrupt_after = nil,  -- mwait raises "Interrupted." after N waits (VLC 3 behaviour)
        osd = {},
        logs = {},
        added = {},             -- add_subtitle calls
        adds_pending = {},
        add_delay_us = 0,       -- delay before an added track shows up in spu-es
        next_es = 100,
        var_sets = {},
        dialogs = {},
        input = nil,
    }

    local function log(level)
        return function(...)
            local parts = {}
            for i = 1, select("#", ...) do parts[#parts + 1] = tostring(select(i, ...)) end
            m.logs[#m.logs + 1] = level .. " " .. table.concat(parts, " ")
        end
    end

    -- materialise pending add_subtitle() calls whose delay elapsed
    local function process_adds()
        if not m.input then return end
        local keep = {}
        for _, a in ipairs(m.adds_pending) do
            if m.now >= a.at then
                local id = m.next_es
                m.next_es = m.next_es + 1
                local n = #m.input.spu + 1
                m.input.spu[n] = { id = id, label = "Track " .. n }
                if a.select then m.input.sel["spu-es"] = id end
            else
                keep[#keep + 1] = a
            end
        end
        m.adds_pending = keep
    end
    m.process_adds = process_adds

    -- tests: set the playing input
    function m.set_input(uri, audio, spu, audio_sel, spu_sel)
        m.input = {
            uri = uri,
            obj = { _input = true },
            audio = audio or {},
            spu = spu or {},
            -- "time" (playback position) and "spu-delay" are integers in microseconds
            sel = { ["audio-es"] = audio_sel or -1, ["spu-es"] = spu_sel or -1,
                    ["time"] = 0, ["spu-delay"] = 0, ["length"] = opts.length or 0 },
        }
        m.adds_pending = {}
    end
    function m.select(var, id)
        m.input.sel[var] = id
    end
    -- tests: playback position in seconds
    function m.set_time(sec)
        m.input.sel["time"] = math.floor(sec * 1000000 + 0.5)
    end
    function m.stop()
        m.input = nil
    end

    local vlc = {}
    vlc.msg = { dbg = log("dbg"), info = log("info"), warn = log("warn"), err = log("err") }
    vlc.misc = {
        mdate = function() return m.now end,
        mwait = function(t)
            m.waits = m.waits + 1
            if m.interrupt_after and m.waits >= m.interrupt_after then
                error("Interrupted.")
            end
            if t > m.now then m.now = t end
        end,
    }
    vlc.config = { userdatadir = function() return opts.userdatadir end }
    vlc.io = { mkdir = opts.mkdir }
    vlc.strings = {
        make_uri = function(p) return "file://" .. p end,
        make_path = opts.make_path,
    }
    vlc.osd = {
        message = function(text, channel, position, duration)
            m.osd[#m.osd + 1] = text
            m.last_osd = { text = text, channel = channel, position = position, duration = duration }
        end,
        channel_register = function() return 7 end,
    }
    vlc.object = {
        input = function() return m.input and m.input.obj or nil end,
    }
    vlc.input = {
        item = function()
            if not m.input then return nil end
            local uri = m.input.uri
            return { uri = function() return uri end }
        end,
        add_subtitle = function(path, autoselect)
            if not m.input then error("can't add subtitle: no current input") end
            m.added[#m.added + 1] = { path = path, select = autoselect }
            m.adds_pending[#m.adds_pending + 1] = { at = m.now + m.add_delay_us, select = autoselect }
            if m.add_delay_us == 0 then process_adds() end
        end,
    }
    vlc.var = {
        get_list = function(obj, name)
            assert(obj and obj._input, "get_list on non-input object")
            process_adds()
            if name == "audio-es" then return copy_tracks(m.input.audio) end
            if name == "spu-es" then return copy_tracks(m.input.spu) end
            error("unknown var " .. tostring(name))
        end,
        get = function(obj, name)
            assert(obj and obj._input, "get on non-input object")
            process_adds()
            return m.input.sel[name]
        end,
        set = function(obj, name, value)
            assert(obj and obj._input, "set on non-input object")
            m.input.sel[name] = value
            m.var_sets[#m.var_sets + 1] = { name = name, value = value }
        end,
    }
    -- extension dialogs
    vlc.dialog = function(title)
        local d = { title = title, widgets = {}, shown = false, deleted = false }
        local function widget(kind, text, cb)
            local w = { kind = kind, text = text, cb = cb }
            function w:set_text(t) self.text = t end
            function w:get_text() return self.text end
            d.widgets[#d.widgets + 1] = w
            return w
        end
        function d:add_html(text) return widget("html", text) end
        function d:add_label(text) return widget("label", text) end
        function d:add_button(text, cb) return widget("button", text, cb) end
        function d:show() self.shown = true end
        function d:update() end
        function d:delete() self.deleted = true end
        function d:button(text)
            for _, w in ipairs(self.widgets) do
                if w.kind == "button" and w.text == text then return w end
            end
        end
        m.dialogs[#m.dialogs + 1] = d
        m.last_dialog = d
        return d
    end

    m.vlc = vlc
    return m
end

return Mock
