//! The ``## Response Format`` block through the server's routes: prompts equal encoding.py's (encoder_unicode.json).
const std = @import("std");
const api = @import("engine_api");
const assets = @import("dsv41_assets");
const server = @import("root.zig");
const json = server.json;
const routes = @import("routes.zig");
const deepseek_schema = @import("deepseek_schema.zig");
const Conn = @import("http_conn.zig").Conn;
const Rig = @import("dsv41_schema_prompt_test.zig").Rig;
const Value = json.Value;

const encoder_json = @embedFile("fixtures/dsv41_response_format/encoder_unicode.json");
const user_tag = "<\u{ff5c}User\u{ff5c}>";

/// The fixture: the schema, its conversation and the encoder's prompts with and without the block.
const Case = struct {
    schema: Value,
    schema_json: []const u8,
    system: []const u8,
    user: []const u8,
    block: []const u8,
    on: []const u8,
    off: []const u8,

    fn load(a: std.mem.Allocator) !Case {
        const v = (try json.parse(a, encoder_json)).ok;
        return .{ .schema = v.get("schema").?, .schema_json = v.get("schema_json").?.string, .system = v.get("system").?.string, .user = v.get("user").?.string, .block = v.get("block").?.string, .on = v.get("prompt_on").?.string, .off = v.get("prompt_off").?.string };
    }
};

/// Records each request it is given (its prompt, shared prefixes and image count) and finishes it at once.
const Capture = struct {
    vision: bool = false,
    prompt: []u32 = &.{},
    shared: []u32 = &.{},
    images: usize = 0,
    submits: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 1 << 16, .structures = true, .vision = if (e.vision) .{ .ctx = e, .placeholder = Rig.image_placeholder, .token = Rig.image_id, .prepare = prepareImage } else null };
    }

    fn prepareImage(_: *anyopaque, _: std.mem.Allocator, bytes: []const u8, _: *[]const u8) api.Vision.PrepareError!api.Vision.Prepared {
        return .{ .tokens = 3, .payload = bytes };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        const gpa = std.testing.allocator;
        gpa.free(e.prompt);
        gpa.free(e.shared);
        e.prompt = gpa.dupe(u32, request.prompt) catch return error.Busy;
        e.shared = gpa.dupe(u32, request.shared_prefixes) catch return error.Busy;
        e.images = request.images.len;
        e.submits += 1;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn deinit(e: *Capture) void {
        std.testing.allocator.free(e.prompt);
        std.testing.allocator.free(e.shared);
    }
};

/// A server over the deployed template with the switch `on`, and what its engine was given.
const Served = struct {
    rig: Rig,
    capture: Capture,
    srv: *server.Server = undefined,

    fn start(s: *Served, a: std.mem.Allocator, on: bool, vision: bool) !void {
        s.rig = try Rig.init(a, assets.chat_template);
        s.capture = .{ .vision = vision };
        s.srv = try server.Server.init(std.testing.allocator, std.testing.io, s.capture.engine(), s.rig.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .enable_thinking = false, .dsv41_schema_prompt = on }, null);
    }

    fn stop(s: *Served) void {
        s.srv.deinit();
        s.capture.deinit();
    }

    /// The whole HTTP reply to one POST, its JSON body parsed.
    fn post(s: *Served, a: std.mem.Allocator, path: []const u8, body: Value) !Value {
        const text = try json.stringify(a, body, .{});
        var pair: [2]std.c.fd_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer _ = std.c.close(pair[0]);
        defer _ = std.c.close(pair[1]);
        const head = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n", .{ path, text.len });
        for ([_][]const u8{ head, text }) |part| {
            var sent: usize = 0;
            while (sent < part.len) {
                const n = std.c.write(pair[0], part[sent..].ptr, part.len - sent);
                if (n <= 0) return error.Closed;
                sent += @intCast(n);
            }
        }
        var conn = try Conn.init(std.testing.allocator, pair[1], "test");
        defer conn.deinit();
        conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
        routes.dispatch(s.srv, &conn, arena.allocator());
        var got: std.ArrayList(u8) = .empty;
        var tmp: [8192]u8 = undefined;
        var pollfd = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
        while (try std.posix.poll(&pollfd, 200) > 0) {
            const n = try std.posix.read(pair[0], &tmp);
            if (n == 0) break;
            try got.appendSlice(a, tmp[0..n]);
        }
        const at = std.mem.indexOf(u8, got.items, "\r\n\r\n") orelse return error.BadReply;
        if (!std.mem.startsWith(u8, got.items, "HTTP/1.1 200")) {
            std.debug.print("{s} answered {s}\n", .{ path, got.items });
            return error.NotOk;
        }
        return switch (try json.parse(a, got.items[at + 4 ..])) {
            .ok => |v| v,
            .err => error.BadJson,
        };
    }

    fn decoded(s: *Served, a: std.mem.Allocator) ![]u8 {
        return s.rig.text().decode(a, s.capture.prompt);
    }
};

fn obj(a: std.mem.Allocator, pairs: anytype) !Value {
    const o = try json.newObject(a);
    inline for (pairs) |p| try o.put(a, p[0], p[1]);
    return .{ .object = o };
}

fn str(s: []const u8) Value {
    return .{ .string = s };
}

fn arr(a: std.mem.Allocator, items: []const Value) !Value {
    return .{ .array = try a.dupe(Value, items) };
}

/// The chat messages of the case (the user turn `content`).
fn chatMessages(a: std.mem.Allocator, c: Case, content: Value) !Value {
    return arr(a, &.{ try obj(a, .{ .{ "role", str("system") }, .{ "content", str(c.system) } }), try obj(a, .{ .{ "role", str("user") }, .{ "content", content } }) });
}

/// A chat ``response_format`` of the case's schema: as an object, or (`as_text`) as its JSON text.
fn chatFormat(a: std.mem.Allocator, c: Case, as_text: bool) !Value {
    const schema = if (as_text) str(c.schema_json) else c.schema;
    return obj(a, .{ .{ "type", str("json_schema") }, .{ "json_schema", try obj(a, .{ .{ "name", str("n") }, .{ "schema", schema } }) } });
}

fn chatBody(a: std.mem.Allocator, c: Case, content: Value, format: ?Value) !Value {
    const o = try json.newObject(a);
    try o.put(a, "model", str("m"));
    try o.put(a, "max_tokens", .{ .int = "2" });
    try o.put(a, "messages", try chatMessages(a, c, content));
    if (format) |f| try o.put(a, "response_format", f);
    return .{ .object = o };
}

/// The ids /tokenize gives the chat body.
fn tokenized(s: *Served, a: std.mem.Allocator, body: Value) ![]u32 {
    const r = try s.post(a, "/tokenize", body);
    const list = r.get("tokens").?.array;
    const ids = try a.alloc(u32, list.len);
    for (list, ids) |t, *id| id.* = @intCast(t.int64().?);
    try std.testing.expectEqual(@as(i64, @intCast(ids.len)), r.get("count").?.int64().?);
    return ids;
}

fn expectPrompt(s: *Served, a: std.mem.Allocator, expected: []const u8) !void {
    try std.testing.expectEqualStrings(expected, try s.decoded(a));
}

test "Chat: the routed prompt is the encoder's, with the block on and without it off, and /tokenize gives the same ids" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    for ([_]bool{ true, false }) |on| {
        var s: Served = undefined;
        try s.start(a, on, false);
        defer s.stop();
        const body = try chatBody(a, c, str(c.user), try chatFormat(a, c, false));
        _ = try s.post(a, "/v1/chat/completions", body);
        try expectPrompt(&s, a, if (on) c.on else c.off);
        try std.testing.expectEqualSlices(u32, s.capture.prompt, try tokenized(&s, a, body));
    }
}

test "Chat: a json_schema schema given as JSON text shows the same block as the object" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    var s: Served = undefined;
    try s.start(a, true, false);
    defer s.stop();
    _ = try s.post(a, "/v1/chat/completions", try chatBody(a, c, str(c.user), try chatFormat(a, c, true)));
    try expectPrompt(&s, a, c.on);
    // and schemaFor itself: the text parsed to the object, which to_json writes as the encoder does
    const from_text = (try deepseek_schema.schemaFor(a, try chatFormat(a, c, true))).?;
    const from_object = (try deepseek_schema.schemaFor(a, try chatFormat(a, c, false))).?;
    try std.testing.expectEqualStrings(try json.stringify(a, from_object, .{ .ascii = false }), try json.stringify(a, from_text, .{ .ascii = false }));
    try std.testing.expect(std.mem.indexOf(u8, try s.decoded(a), c.block) != null);
}

test "Messages: the routed prompt is the encoder's on and off, and count_tokens counts it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    for ([_]bool{ true, false }) |on| {
        var s: Served = undefined;
        try s.start(a, on, false);
        defer s.stop();
        const body = try obj(a, .{
            .{ "model", str("m") },
            .{ "max_tokens", Value{ .int = "2" } },
            .{ "system", str(c.system) },
            .{ "messages", try arr(a, &.{try obj(a, .{ .{ "role", str("user") }, .{ "content", str(c.user) } })}) },
            .{ "output_config", try obj(a, .{.{ "format", try obj(a, .{ .{ "type", str("json_schema") }, .{ "schema", c.schema } }) }}) },
        });
        _ = try s.post(a, "/v1/messages", body);
        try expectPrompt(&s, a, if (on) c.on else c.off);
        const counted = try s.post(a, "/v1/messages/count_tokens", body);
        try std.testing.expectEqual(@as(i64, @intCast(s.capture.prompt.len)), counted.get("input_tokens").?.int64().?);
    }
}

test "Responses: the routed prompt is the encoder's on and off, and the chat body's /tokenize gives the same ids" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    for ([_]bool{ true, false }) |on| {
        var s: Served = undefined;
        try s.start(a, on, false);
        defer s.stop();
        const body = try obj(a, .{
            .{ "model", str("m") },
            .{ "max_output_tokens", Value{ .int = "16" } },
            .{ "instructions", str(c.system) },
            .{ "input", str(c.user) },
            .{ "text", try obj(a, .{.{ "format", try obj(a, .{ .{ "type", str("json_schema") }, .{ "name", str("n") }, .{ "schema", c.schema } }) }}) },
        });
        _ = try s.post(a, "/v1/responses", body);
        try expectPrompt(&s, a, if (on) c.on else c.off);
        try std.testing.expectEqualSlices(u32, s.capture.prompt, try tokenized(&s, a, try chatBody(a, c, str(c.user), try chatFormat(a, c, false))));
    }
}

test "the system-prefix probe shares the system message with its block: the reusable prefix ends at the user turn" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    try std.testing.expect(c.system.len > 512);
    for ([_]bool{ true, false }) |on| {
        var s: Served = undefined;
        try s.start(a, on, false);
        defer s.stop();
        _ = try s.post(a, "/v1/chat/completions", try chatBody(a, c, str(c.user), try chatFormat(a, c, false)));
        const text = try s.decoded(a);
        // byte ids: the prefix the probe shares is the system message (with the block when on) and the user tag
        const user_at = std.mem.indexOf(u8, text, user_tag).? + user_tag.len;
        try std.testing.expect(s.capture.shared.len > 0);
        try std.testing.expectEqual(@as(u32, @intCast(user_at)), s.capture.shared[s.capture.shared.len - 1]);
        if (on) try std.testing.expect(std.mem.indexOf(u8, text[0..user_at], c.block) != null);
    }
}

test "an image request: the routed prompt holds the image's span, the block on and not off, as /tokenize gives it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    const content = try arr(a, &.{
        try obj(a, .{ .{ "type", str("text") }, .{ "text", str(c.user) } }),
        try obj(a, .{ .{ "type", str("image_url") }, .{ "image_url", try obj(a, .{.{ "url", str("data:image/png;base64,iVBORw0KGgo=") }}) } }),
    });
    for ([_]bool{ true, false }) |on| {
        var s: Served = undefined;
        try s.start(a, on, true);
        defer s.stop();
        const body = try chatBody(a, c, content, try chatFormat(a, c, false));
        _ = try s.post(a, "/v1/chat/completions", body);
        try std.testing.expectEqual(@as(usize, 1), s.capture.images);
        try std.testing.expectEqual(@as(usize, 3), std.mem.count(u32, s.capture.prompt, &.{Rig.image_id}));
        const text = try s.decoded(a);
        try std.testing.expectEqual(on, std.mem.indexOf(u8, text, c.block) != null);
        // the system message is the encoder's either way
        const sys_end = std.mem.indexOf(u8, text, user_tag).?;
        const want = if (on) c.on else c.off;
        try std.testing.expectEqualStrings(want[0..std.mem.indexOf(u8, want, user_tag).?], text[0..sys_end]);
        try std.testing.expectEqualSlices(u32, s.capture.prompt, try tokenized(&s, a, body));
    }
}

test "an image request whose schema holds the image marker as text: the marker stays text, the image its one span" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    const content = try arr(a, &.{
        try obj(a, .{ .{ "type", str("text") }, .{ "text", str(c.user) } }),
        try obj(a, .{ .{ "type", str("image_url") }, .{ "image_url", try obj(a, .{.{ "url", str("data:image/png;base64,iVBORw0KGgo=") }}) } }),
    });
    const marker = "the literal marker " ++ Rig.image_placeholder;
    const schema = try obj(a, .{ .{ "type", str("object") }, .{ "description", str(marker) }, .{ "properties", try obj(a, .{.{ Rig.image_placeholder, try obj(a, .{.{ "type", str("string") }}) }}) } });
    const format = try obj(a, .{ .{ "type", str("json_schema") }, .{ "json_schema", try obj(a, .{ .{ "name", str("n") }, .{ "schema", schema } }) } });
    var s: Served = undefined;
    try s.start(a, true, true);
    defer s.stop();
    const body = try chatBody(a, c, content, format);
    _ = try s.post(a, "/v1/chat/completions", body);
    try std.testing.expectEqual(@as(usize, 1), s.capture.images);
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u32, s.capture.prompt, &.{Rig.image_id}));
    // the schema's markers, in a value and a key, shown as plain text (ASCII bars), as a message's own text is
    const text = try s.decoded(a);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, text, "<|deepseek_image|>"));
    try std.testing.expectEqualSlices(u32, s.capture.prompt, try tokenized(&s, a, body));
}

test "shared_messages: the client's cut carries the system message's block and ends at the next user turn, on and off" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    try std.testing.expect(c.system.len > 512);
    for ([_]bool{ true, false }) |on| {
        var s: Served = undefined;
        try s.start(a, on, false);
        defer s.stop();
        const messages = try arr(a, &.{
            try obj(a, .{ .{ "role", str("system") }, .{ "content", str(c.system) } }),
            try obj(a, .{ .{ "role", str("user") }, .{ "content", str(c.user) } }),
            try obj(a, .{ .{ "role", str("assistant") }, .{ "content", str("{\"ok\": true}") } }),
            try obj(a, .{ .{ "role", str("user") }, .{ "content", str("And again, please.") } }),
        });
        const body = try obj(a, .{ .{ "model", str("m") }, .{ "max_tokens", Value{ .int = "2" } }, .{ "messages", messages }, .{ "response_format", try chatFormat(a, c, false) }, .{ "shared_messages", Value{ .int = "3" } } });
        _ = try s.post(a, "/v1/chat/completions", body);
        const text = try s.decoded(a);
        // byte ids: the first three messages' prefix runs to the last user turn's tag, where the probe turn differs
        const cut = std.mem.lastIndexOf(u8, text, user_tag).? + user_tag.len;
        try std.testing.expect(s.capture.shared.len > 0);
        try std.testing.expectEqual(@as(u32, @intCast(cut)), s.capture.shared[s.capture.shared.len - 1]);
        try std.testing.expectEqual(on, std.mem.indexOf(u8, text[0..cut], c.block) != null);
    }
}

/// `off` with the case's block where its first user turn starts: the end of the first system message.
fn withBlock(a: std.mem.Allocator, c: Case, off: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, off, user_tag).?;
    return std.mem.concat(a, u8, &.{ off[0..at], c.block, off[at..] });
}

test "Chat with tools and no system message: the block follows the tools on the empty system message the tools add" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    const params = try obj(a, .{ .{ "type", str("object") }, .{ "properties", try obj(a, .{.{ "q", try obj(a, .{.{ "type", str("string") }}) }}) } });
    const tool = try obj(a, .{ .{ "type", str("function") }, .{ "function", try obj(a, .{ .{ "name", str("lookup") }, .{ "description", str("Looks a word up.") }, .{ "parameters", params } }) } });
    var prompts: [2][]u8 = undefined;
    for ([_]bool{ true, false }, 0..) |on, i| {
        var s: Served = undefined;
        try s.start(a, on, false);
        defer s.stop();
        const body = try obj(a, .{
            .{ "model", str("m") },
            .{ "max_tokens", Value{ .int = "2" } },
            .{ "messages", try arr(a, &.{try obj(a, .{ .{ "role", str("user") }, .{ "content", str(c.user) } })}) },
            .{ "tools", try arr(a, &.{tool}) },
            .{ "response_format", try chatFormat(a, c, false) },
        });
        _ = try s.post(a, "/v1/chat/completions", body);
        prompts[i] = try s.decoded(a);
        try std.testing.expectEqualSlices(u32, s.capture.prompt, try tokenized(&s, a, body));
    }
    const sys_open = "<\u{ff5c}System\u{ff5c}>\n\n";
    try std.testing.expect(std.mem.indexOf(u8, prompts[1], sys_open) != null);
    try std.testing.expect(std.mem.indexOf(u8, prompts[1], "lookup") != null);
    try std.testing.expectEqualStrings(try withBlock(a, c, prompts[1]), prompts[0]);
}

test "Chat with thinking on: the block follows the effort prompt and the system text, as off without it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    var prompts: [2][]u8 = undefined;
    for ([_]bool{ true, false }, 0..) |on, i| {
        var s: Served = undefined;
        try s.start(a, on, false);
        defer s.stop();
        const body = try chatBody(a, c, str(c.user), try chatFormat(a, c, false));
        try body.object.put(a, "chat_template_kwargs", try obj(a, .{.{ "enable_thinking", Value{ .bool = true } }}));
        _ = try s.post(a, "/v1/chat/completions", body);
        prompts[i] = try s.decoded(a);
        try std.testing.expectEqualSlices(u32, s.capture.prompt, try tokenized(&s, a, body));
    }
    try std.testing.expect(std.mem.indexOf(u8, prompts[1], "Reasoning Effort: ") != null);
    try std.testing.expectEqualStrings(try withBlock(a, c, prompts[1]), prompts[0]);
}

test "a text-only request whose schema holds the image marker as text: the marker stays text, with or without vision" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Case.load(a);
    const marker = "the literal marker " ++ Rig.image_placeholder;
    const schema = try obj(a, .{ .{ "type", str("object") }, .{ "description", str(marker) }, .{ "properties", try obj(a, .{.{ Rig.image_placeholder, try obj(a, .{.{ "type", str("string") }}) }}) } });
    const format = try obj(a, .{ .{ "type", str("json_schema") }, .{ "json_schema", try obj(a, .{ .{ "name", str("n") }, .{ "schema", schema } }) } });
    for ([_]bool{ true, false }) |vision| {
        var s: Served = undefined;
        try s.start(a, true, vision);
        defer s.stop();
        const body = try chatBody(a, c, str(c.user), format);
        _ = try s.post(a, "/v1/chat/completions", body);
        try std.testing.expectEqual(@as(usize, 0), s.capture.images);
        // no image token: the schema's markers, in a value and a key, shown as plain text (ASCII bars)
        try std.testing.expectEqual(@as(usize, 0), std.mem.count(u32, s.capture.prompt, &.{Rig.image_id}));
        const text = try s.decoded(a);
        try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, text, "<|deepseek_image|>"));
        try std.testing.expectEqualSlices(u32, s.capture.prompt, try tokenized(&s, a, body));
        // the system-prefix probe renders the same escaped block: the reusable prefix still ends at the user turn
        const user_at = std.mem.indexOf(u8, text, user_tag).? + user_tag.len;
        try std.testing.expectEqual(@as(u32, @intCast(user_at)), s.capture.shared[s.capture.shared.len - 1]);
    }
}

test "a system message later in the conversation stays in place: the block goes on an empty system message first" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = (try json.parse(a, @embedFile("fixtures/dsv41_response_format/encoder_late_system.json"))).ok;
    const format = try obj(a, .{ .{ "type", str("json_schema") }, .{ "json_schema", try obj(a, .{ .{ "name", str("n") }, .{ "schema", v.get("schema").? } }) } });
    for ([_]bool{ true, false }) |on| {
        var s: Served = undefined;
        try s.start(a, on, false);
        defer s.stop();
        const body = try obj(a, .{ .{ "model", str("m") }, .{ "max_tokens", Value{ .int = "2" } }, .{ "messages", v.get("messages").? }, .{ "response_format", format } });
        _ = try s.post(a, "/v1/chat/completions", body);
        try expectPrompt(&s, a, v.get(if (on) "prompt_on" else "prompt_off").?.string);
        try std.testing.expectEqualSlices(u32, s.capture.prompt, try tokenized(&s, a, body));
    }
}
