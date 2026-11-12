//! Prompt ids from a request, as ``prepare_prompt`` and ``render_prompt_ids`` make them.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const api = @import("engine_api");
const deepseek_schema = @import("deepseek_schema.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const Rendered = struct { ids: []const u32, history_len: usize = 0, images: []const api.Image = &.{} };

pub const isTitle = messages_mod.isTitleRequest;

/// The schema a request's ``response_format`` shows in a DeepSeek-V4.1 prompt (its ``## Response Format`` block), or null:
/// always null unless TF_DSV41_SCHEMA_PROMPT=1 (Config.dsv41_schema_prompt) and the template reads the V4.1 dialect.
pub fn schemaOf(srv: *const Server, a: std.mem.Allocator, response_format: ?Value) std.mem.Allocator.Error!?Value {
    if (!srv.config.dsv41_schema_prompt or srv.dialect != .deepseek_v41) return null;
    return deepseek_schema.schemaFor(a, response_format orelse return null);
}

/// ``render_prompt_ids``: messages normalized for this template, rendered, and a dangling ``<think>`` closed. `schema`:
/// schemaOf's, or null.
pub fn renderIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool, schema: ?Value) errors.Refused![]const u32 {
    const normalized = try messages_mod.normalize(cx, messages, srv.late_system, srv.needs_user_after_tool, srv.dialect);
    return renderNormalizedIds(srv, cx, normalized, tools, thinking, effort, generation, schema);
}

/// Render normalized messages without rejecting authenticated image placeholders a second time.
fn renderNormalizedIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool, schema: ?Value) errors.Refused![]const u32 {
    const normalized = try messages_mod.toolArguments(cx, messages, srv.dialect);
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
        .reasoning_effort_value = if (thinking) srv.effortValue(effort) else null,
        .response_format = schema,
    };
    var ids = srv.text.renderIds(cx.a, normalized, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "{s}", .{problem}),
    };
    if (!thinking and generation and ids.len > 0) {
        const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
        if (std.mem.eql(u8, @import("reply_text.zig").pyStrip(last), "<think>")) {
            const close = srv.text.encode(cx.a, "</think>", false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => &[_]u32{},
            };
            if (close.len == 1) ids = try std.mem.concat(cx.a, u32, &.{ ids, close });
        }
    }
    return ids;
}

/// A request's prompt: raw text or ids for a completion, else the chat template's, with its history length.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    if (input.prompt) |p| switch (p) {
        .text => |t| return .{ .ids = srv.text.encode(cx.a, t, true) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
        } },
        .ids => |ids| return .{ .ids = ids },
    };
    const schema = try schemaOf(srv, cx.a, input.fields.get("response_format"));
    if (input.images.len > 0) return renderImages(srv, cx, input.messages, input.tools, thinking, effort, true, input.images, schema);
    const prompt = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, true, schema);
    const history = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, false, schema);
    var history_len: usize = 0;
    if (history.len > 0 and history.len < prompt.len and std.mem.eql(u32, prompt[0..history.len], history)) history_len = history.len;
    if (history_len == 0 and prompt.len > 1 and std.mem.eql(u32, history, prompt)) history_len = prompt.len - 1; // a template with no generation suffix
    return .{ .ids = prompt, .history_len = history_len };
}

/// Render authenticated image messages after normalizeImages has escaped literal placeholders.
pub fn renderImages(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool, images: []const []const u8, schema: ?Value) errors.Refused!Rendered {
    const vision = srv.info.vision orelse return cx.refuse("image input: this engine has no vision hook");
    const tagged = try imageIdentities(cx, messages, vision.placeholder, images.len);
    const ids = try renderNormalizedIds(srv, cx, tagged, tools, thinking, effort, generation, schema);
    return spans(srv, cx, ids, images);
}

/// Identity travels inside paired image delimiters until rendering has established the final order.
fn imageIdentities(cx: *Cx, messages: Value, placeholder: []const u8, count: usize) errors.Refused!Value {
    if (placeholder.len == 0) return cx.refuse("image input: empty image placeholder");
    const tagged = try cx.a.dupe(Value, messages.array);
    var index: usize = 0;
    for (tagged) |*message| {
        const content = message.get("content") orelse continue;
        if (content != .string) continue;
        var text: std.ArrayList(u8) = .empty;
        var from: usize = 0;
        while (std.mem.indexOfPos(u8, content.string, from, placeholder)) |at| {
            try text.appendSlice(cx.a, content.string[from..at]);
            try text.appendSlice(cx.a, try std.fmt.allocPrint(cx.a, "{s}{d}{s}", .{ placeholder, index, placeholder }));
            index += 1;
            from = at + placeholder.len;
        }
        if (from == 0) continue;
        try text.appendSlice(cx.a, content.string[from..]);
        const copy = try json.copyObject(cx.a, message.object);
        try copy.put(cx.a, "content", .{ .string = text.items });
        message.* = .{ .object = copy };
    }
    if (index != count) return cx.refuse("image input: normalized placeholders do not match the image inputs");
    return .{ .array = tagged };
}

/// Each image placeholder of the rendered prompt as its image's span (the vision family's prepare: the span's length and
/// the payload the engine reads), the images in prompt order.
fn spans(srv: *Server, cx: *Cx, ids: []const u32, images: []const []const u8) errors.Refused!Rendered {
    const vision = srv.info.vision orelse return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
    var out: std.ArrayList(u32) = .empty;
    var placed: std.ArrayList(api.Image) = .empty;
    const used = try cx.a.alloc(bool, images.len);
    @memset(used, false);
    var cursor: usize = 0;
    var tokens: u64 = 0;
    var pixels: u64 = 0;
    while (cursor < ids.len) {
        const id = ids[cursor];
        if (id != vision.token) {
            try out.append(cx.a, id);
            cursor += 1;
            continue;
        }
        const end = std.mem.indexOfScalarPos(u32, ids, cursor + 1, vision.token) orelse
            return cx.refuse("image input: the template removed an image identity delimiter");
        const identity = try srv.text.decode(cx.a, ids[cursor + 1 .. end]);
        const k = std.fmt.parseInt(usize, identity, 10) catch
            return cx.refuse("image input: the template changed an image identity");
        if (k >= images.len or used[k]) return cx.refuse("image input: unknown or repeated image identity");
        used[k] = true;
        cursor = end + 1;
        var problem: []const u8 = "";
        const prep = vision.prepare(vision.ctx, cx.a, images[k], &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.BadImage => return cx.refuse(problem),
        };
        try placed.append(cx.a, .{ .at = @intCast(out.items.len), .tokens = prep.tokens, .bytes = prep.payload });
        try out.appendNTimes(cx.a, vision.token, prep.tokens);
        // the limits refuse at the picture that crosses them, before any later one is decoded
        pixels += prep.pixels;
        tokens += prep.tokens;
        const n = placed.items.len;
        if (vision.max_pixels > 0 and pixels > vision.max_pixels) return cx.fail(.request, "image input: the images decode to {d} pixels by image {d} of {d}, more than the {d} this engine takes per request; send fewer or smaller images", .{ pixels, n, images.len, vision.max_pixels });
        if (vision.max_tokens > 0 and tokens > vision.max_tokens) return cx.fail(.request, "image input: the images take {d} image tokens by image {d} of {d}, more than the {d} this engine holds per request; send fewer or smaller images", .{ tokens, n, images.len, vision.max_tokens });
    }
    if (placed.items.len != images.len) return cx.refuse("image input: the prompt holds fewer image placeholders than images");
    return .{ .ids = out.items, .images = placed.items };
}

/// A reusable system prefix, found with a probe in place of the first user message; zero below 512 tokens. `schema`: the
/// prompt's own (schemaOf), so the probe's system block is the prompt's.
pub fn systemPrefixLen(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8, schema: ?Value) usize {
    if (messages != .array) return 0;
    const first_user = for (messages.array, 0..) |m, i| {
        const role = m.get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break i;
    } else return 0;
    return probedPrefix(srv, cx, messages, first_user, tools, prompt_ids, thinking, effort, schema);
}

/// The tokens the first `k` messages render to as a prefix of `prompt_ids` (`shared_messages`); 0 outside or under 512.
pub fn messagesPrefixLen(srv: *Server, cx: *Cx, messages: Value, k: usize, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8, schema: ?Value) usize {
    if (messages != .array or k == 0 or k >= messages.array.len) return 0;
    return probedPrefix(srv, cx, messages, k, tools, prompt_ids, thinking, effort, schema);
}

/// The first `n` messages and a probe user turn rendered: how far their ids agree with the prompt's (under 512: 0).
fn probedPrefix(srv: *Server, cx: *Cx, messages: Value, n: usize, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8, schema: ?Value) usize {
    const probe = cx.a.alloc(Value, n + 1) catch return 0;
    @memcpy(probe[0..n], messages.array[0..n]);
    const user = json.newObject(cx.a) catch return 0;
    user.put(cx.a, "role", .{ .string = "user" }) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    probe[n] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const other = renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, effort, true, schema) catch return 0;
    var shared: usize = 0;
    while (shared < @min(prompt_ids.len, other.len) and prompt_ids[shared] == other[shared]) shared += 1;
    return if (shared >= 512) shared else 0;
}
