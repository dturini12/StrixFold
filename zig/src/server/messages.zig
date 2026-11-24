//! Chat messages as the template sees them: leading instructions merged, text parts joined, call arguments as objects.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const Value = json.Value;
const Cx = errors.Cx;

const roles = [_][]const u8{ "system", "developer", "user", "assistant", "tool" };

/// How a chat template wants a request's messages prepared. ``generic``: the server's rules for every family (text
/// parts concatenated, the five OpenAI roles, a non-empty list, call arguments decoded once). ``deepseek_v41``: the
/// input rules of DeepSeek-V4.1's own encoder (``encode_messages`` in encoding/encoding.py at 1bae0e10) for a template
/// written from it, which renders what those rules let through block by block as the encoder does: text parts joined
/// by a blank line, the ``latest_reminder`` role, ``tool_result`` parts in a user message, an empty conversation (BOS
/// alone), every system message its own block (leading ones are not merged), null content as the encoder writes it,
/// call arguments decoded up to twice, and the image placeholder refused in text.
pub const Dialect = enum { generic, deepseek_v41 };

/// DeepSeek-V4.1's image placeholder (``IMAGE_PLACEHOLDER``): one special token, so text holding it would put an
/// image token into a text-only prompt.
pub const deepseek_image = "<\u{ff5c}deepseek_image\u{ff5c}>";
pub const deepseek_image_refusal = "message text must not contain the image placeholder <\u{ff5c}deepseek_image\u{ff5c}> (images are content parts, and this server accepts text only)";

/// The dialect a chat template asks for: DeepSeek-V4.1's when it renders that encoder's latest_reminder token and its
/// DSML invoke tags.
pub fn dialectOf(source: []const u8) Dialect {
    const v41 = std.mem.indexOf(u8, source, "<\u{ff5c}latest_reminder\u{ff5c}>") != null and
        std.mem.indexOf(u8, source, "\u{ff5c}DSML\u{ff5c} invoke") != null;
    return if (v41) .deepseek_v41 else .generic;
}

fn isRole(role: ?Value, dialect: Dialect) bool {
    const r = role orelse return false;
    if (r != .string) return false;
    for (roles) |name| if (std.mem.eql(u8, name, r.string)) return true;
    return dialect == .deepseek_v41 and std.mem.eql(u8, r.string, "latest_reminder");
}

fn hasImageToken(s: []const u8) bool {
    return std.mem.indexOf(u8, s, deepseek_image) != null;
}

/// A text part's text: the part's ``text`` string, the image placeholder refused (``_process_image_blocks``).
fn dsv41PartText(cx: *Cx, part: Value) errors.Refused![]const u8 {
    const t = part.get("text") orelse return cx.refuse("a text content part must contain a text string");
    if (t != .string) return cx.refuse("a text content part must contain a text string");
    if (hasImageToken(t.string)) return cx.refuse(deepseek_image_refusal);
    return t.string;
}

/// A ``tool_result`` part of a user message as ``render_message`` reads it: its content text, or a list whose text parts
/// it joins and whose other parts it shows as ``[Unsupported type]``; images refused (text only), the placeholder
/// refused in text parts. Its string content is not checked for the placeholder, as the encoder does not check it.
/// Returns whether its content is null, which the encoder writes as the text None.
fn dsv41ToolResult(cx: *Cx, part: Value) errors.Refused!bool {
    const content = part.get("content") orelse return false;
    if (content == .null) return true;
    if (content != .array) return false;
    for (content.array) |inner| {
        if (inner != .object) return cx.refuse("each content part must be an object");
        if (isImagePart(inner) or fields.hasMedia(inner)) return cx.refuse(text_only_refusal);
        const t = inner.get("type");
        if (t != null and t.? == .string and std.mem.eql(u8, t.?.string, "text")) _ = try dsv41PartText(cx, inner);
    }
    return false;
}

/// One message as ``encode_messages`` takes it on a text-only request. The image placeholder is refused in its content
/// and reasoning_content text (``_validate_no_image_sp_tokens``) and in text parts. A list of parts becomes its text
/// parts joined by a blank line (``process_image_messages``), except that a user or tool message holding other parts
/// (tool results, unsupported types) keeps its list for the template, which renders it block by block as
/// ``render_message`` does; other roles' non-text parts are dropped, as the encoder's join drops them. Null and absent
/// content as the encoder writes them: a user message's null content is refused (the encoder's join fails on it); a
/// tool result's null content and a reminder's null or absent content are the text None (``str.format`` of None);
/// anything else is an empty string.
fn dsv41Message(cx: *Cx, obj: *json.Object, role: []const u8) errors.Refused!*json.Object {
    if (obj.get("reasoning_content")) |r| if (r == .string and hasImageToken(r.string)) return cx.refuse(deepseek_image_refusal);
    const reminder = std.mem.eql(u8, role, "latest_reminder");
    const content = obj.get("content") orelse return withField(cx, obj, "content", .{ .string = if (reminder) "None" else "" });
    switch (content) {
        .null => {
            if (std.mem.eql(u8, role, "user")) return cx.refuse("a user message's content must be text or a list of parts, not null");
            return withField(cx, obj, "content", .{ .string = if (reminder or std.mem.eql(u8, role, "tool")) "None" else "" });
        },
        .string => |s| {
            if (hasImageToken(s)) return cx.refuse(deepseek_image_refusal);
            return obj;
        },
        .array => |parts| {
            const blocks = std.mem.eql(u8, role, "user") or std.mem.eql(u8, role, "tool");
            var text: std.ArrayList(u8) = .empty;
            var texts: usize = 0;
            var keep = false;
            var null_results: std.ArrayList(usize) = .empty;
            for (parts, 0..) |part, k| {
                if (part != .object) return cx.refuse("each content part must be an object");
                if (isImagePart(part) or fields.hasMedia(part)) return cx.refuse(text_only_refusal);
                const t = part.get("type") orelse return cx.refuse("each content part must have a type");
                if (t != .string) return cx.refuse("each content part must have a type");
                if (std.mem.eql(u8, t.string, "text")) {
                    if (texts > 0) try text.appendSlice(cx.a, "\n\n");
                    try text.appendSlice(cx.a, try dsv41PartText(cx, part));
                    texts += 1;
                } else if (blocks) {
                    if (std.mem.eql(u8, role, "user") and std.mem.eql(u8, t.string, "tool_result") and try dsv41ToolResult(cx, part)) try null_results.append(cx.a, k);
                    keep = true;
                }
            }
            if (keep) {
                if (null_results.items.len == 0) return obj;
                // a tool result's null content is the text None to the encoder; the template would write nothing
                const copy = try cx.a.dupe(Value, parts);
                for (null_results.items) |k| copy[k] = .{ .object = try withField(cx, parts[k].object, "content", .{ .string = "None" }) };
                return withField(cx, obj, "content", .{ .array = copy });
            }
            return withField(cx, obj, "content", .{ .string = text.items });
        },
        else => return cx.refuse("message content must be text or an array of text parts"),
    }
}

fn withField(cx: *Cx, o: *const json.Object, key: []const u8, value: Value) !*json.Object {
    const copy = try json.copyObject(cx.a, o);
    try copy.put(cx.a, key, value);
    return copy;
}

/// A request's images as normalize meets them when the server takes images (its engine's Info.vision): each image
/// part's bytes in prompt order, the part itself the placeholder in its message's text.
pub const Images = struct {
    placeholder: []const u8,
    max: u32, // image parts a request may carry (--vision-max-images), counted before any is decoded
    list: std.ArrayList([]const u8) = .empty,
};

pub const image_text_refusal = "image input: message text must not contain the image placeholder token";

pub const image_role_refusal = "image input: images are accepted only in user messages and tool results";
const text_only_refusal = "this server accepts text parts only; image, audio and video inputs are unsupported";

fn isImagePart(part: Value) bool {
    if (part != .object) return false;
    const t = part.get("type") orelse return false;
    return t == .string and std.mem.eql(u8, t.string, "image_url");
}

/// `s` with every `ph` as plain text: the placeholder's fullwidth bars to ASCII `|`, so the text no longer
/// matches the tokenizer's special (whose id would otherwise add image tokens the prompt's span count misses).
pub fn escape(cx: *Cx, s: []const u8, ph: []const u8) ![]const u8 {
    if (ph.len == 0 or std.mem.indexOf(u8, s, ph) == null) return s;
    var plain: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < ph.len) {
        const n = std.unicode.utf8ByteSequenceLength(ph[i]) catch 1;
        const cp = std.unicode.utf8Decode(ph[i..][0..n]) catch 0;
        if (cp == 0xff5c) try plain.append(cx.a, '|') else try plain.appendSlice(cx.a, ph[i..][0..n]);
        i += n;
    }
    var out: std.ArrayList(u8) = .empty;
    i = 0;
    while (std.mem.indexOfPos(u8, s, i, ph)) |at| {
        try out.appendSlice(cx.a, s[i..at]);
        try out.appendSlice(cx.a, plain.items);
        i = at + ph.len;
    }
    try out.appendSlice(cx.a, s[i..]);
    return out.items;
}

/// The image parts of the messages' content lists.
fn imageCount(list: Value) usize {
    var n: usize = 0;
    for (list.array) |message| {
        if (message != .object) continue;
        const content = message.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |part| n += @intFromBool(isImagePart(part));
    }
    return n;
}

/// An ``image_url`` part's bytes: its URL (the part's string, or its object's ``url``) a base64 data URL, the base64 as
/// Python's b64decode reads it; the served lane's refusals otherwise.
fn imageBytes(cx: *Cx, part: Value) errors.Refused![]const u8 {
    const url: []const u8 = blk: {
        const r = part.get("image_url") orelse break :blk "";
        if (r == .string) break :blk r.string;
        if (r == .object) if (r.get("url")) |u| if (u == .string) break :blk u.string;
        break :blk "";
    };
    if (url.len == 0) return cx.refuse("image input: an image_url part needs a url");
    if (std.mem.startsWith(u8, url, "http://") or std.mem.startsWith(u8, url, "https://")) return cx.refuse("image input: image URLs are off on this server (send a base64 data URL)");
    if (!std.mem.startsWith(u8, url, "data:")) return cx.refuse("image input: image_url must be a data: URL");
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse url.len;
    if (std.mem.indexOf(u8, url[0..comma], ";base64") == null) return cx.refuse("image input: image data URLs must be base64");
    const data = if (comma < url.len) url[comma + 1 ..] else "";
    return switch (b64decode(cx.a, data)) {
        .ok => |b| b,
        .short => |n| cx.fail(.request, "image input: Invalid base64-encoded string: number of data characters ({d}) cannot be 1 more than a multiple of 4", .{n}),
        .padding => cx.refuse("image input: Incorrect padding"),
        .oom => error.OutOfMemory,
    };
}

const B64 = union(enum) { ok: []u8, short: usize, padding, oom };

/// binascii.a2b_base64 as base64.b64decode calls it (not strict): characters outside the alphabet skipped, the input
/// read until a complete padding, an incomplete last quad an error (one character over: "short", with the data
/// characters it counts; two or three: "padding").
fn b64decode(a: std.mem.Allocator, s: []const u8) B64 {
    const out = a.alloc(u8, s.len / 4 * 3 + 3) catch return .oom;
    var n: usize = 0;
    var quad: u8 = 0;
    var left: u8 = 0;
    var pads: usize = 0;
    for (s) |ch| {
        if (ch == '=') {
            if (quad >= 2) {
                pads += 1;
                if (quad + pads >= 4) return .{ .ok = out[0..n] };
            }
            continue;
        }
        const v: u8 = switch (ch) {
            'A'...'Z' => ch - 'A',
            'a'...'z' => ch - 'a' + 26,
            '0'...'9' => ch - '0' + 52,
            '+' => 62,
            '/' => 63,
            else => continue,
        };
        pads = 0;
        switch (quad) {
            0 => {
                quad = 1;
                left = v;
            },
            1 => {
                quad = 2;
                out[n] = (left << 2) | (v >> 4);
                n += 1;
                left = v & 0x0f;
            },
            2 => {
                quad = 3;
                out[n] = (left << 4) | (v >> 2);
                n += 1;
                left = v & 0x03;
            },
            else => {
                quad = 0;
                out[n] = (left << 6) | v;
                n += 1;
                left = 0;
            },
        }
    }
    if (quad == 1) return .{ .short = n / 3 * 4 + 1 };
    if (quad != 0) return .padding;
    return .{ .ok = out[0..n] };
}

test "base64 as Python's b64decode reads it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("hello", b64decode(a, "aGVsbG8=").ok);
    try std.testing.expectEqualStrings("hello", b64decode(a, "aGVs\nbG8=").ok); // a newline skipped
    try std.testing.expectEqualStrings("hello", b64decode(a, "aGVsbG8=ignored").ok); // stops at the padding
    try std.testing.expectEqual(@as(usize, 9), b64decode(a, "aGVsbG8hZ").short); // 9 data characters
    try std.testing.expect(b64decode(a, "aGVsbG8") == .padding);
    try std.testing.expectEqualStrings("", b64decode(a, "").ok);
}

/// ``normalize_messages`` (text only): leading system and developer text merged, later ones as ``late_system``; a template that needs a user query gains one user turn after a trailing tool run; each message's content as `dialect` wants it.
pub fn normalize(cx: *Cx, messages: ?Value, late_system: []const u8, needs_user_after_tool: bool, dialect: Dialect) errors.Refused!Value {
    return normalizeImages(cx, messages, late_system, needs_user_after_tool, dialect, null);
}

/// normalize, and with `images` (a server that takes them) a request with an image part anywhere as the served lane
/// renders it (the reference's process_image_messages): every message's content list joined with blank lines, each
/// image part its placeholder (its bytes into `images`), no message text holding the placeholder.
pub fn normalizeImages(cx: *Cx, messages: ?Value, late_system: []const u8, needs_user_after_tool: bool, dialect: Dialect, images: ?*Images) errors.Refused!Value {
    const list = messages orelse return cx.refuse("messages must be a non-empty list");
    if (list != .array) return cx.refuse("messages must be a non-empty list");
    // encode_messages renders an empty conversation as BOS alone
    if (list.array.len == 0) return if (dialect == .deepseek_v41) .{ .array = try cx.a.alloc(Value, 0) } else cx.refuse("messages must be a non-empty list");
    const count = if (images != null) imageCount(list) else 0;
    if (count > 0 and count > images.?.max) return cx.fail(.request, "image input: the request carries {d} images (its earlier turns' included), more than the {d} this server takes in one request (--vision-max-images)", .{ count, images.?.max });
    const with_images = count > 0;
    var out: std.ArrayList(Value) = .empty;
    var instructions: std.ArrayList(*json.Object) = .empty;
    for (list.array) |message| {
        if (message != .object) return cx.refuse("each message must be an object");
        const role = message.get("role");
        if (!isRole(role, dialect)) return cx.refuse(if (dialect == .deepseek_v41) "message role must be system, developer, user, assistant, tool or latest_reminder" else "message role must be system, developer, user, assistant or tool");
        if (with_images) {
            const im = images.?;
            // a message's own text holding the placeholder is escaped to plain text: its special id would
            // otherwise add image tokens the prompt's span count does not know
            var obj: *json.Object = message.object;
            for ([_][]const u8{ "content", "reasoning_content" }) |key| if (obj.get(key)) |v| if (v == .string and std.mem.indexOf(u8, v.string, im.placeholder) != null) {
                obj = try withField(cx, obj, key, .{ .string = try escape(cx, v.string, im.placeholder) });
            };
            // media other than the content's images is refused as ever
            const rest = try json.copyObject(cx.a, obj);
            _ = rest.orderedRemove("content");
            if (fields.hasMedia(.{ .object = rest })) return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
            const content = obj.get("content");
            var item: *json.Object = obj;
            if (content != null and content.? == .array) {
                const r = role.?.string;
                if (!std.mem.eql(u8, r, "user") and !std.mem.eql(u8, r, "tool")) for (content.?.array) |part| if (isImagePart(part)) return cx.refuse(image_role_refusal);
                var text: std.ArrayList(u8) = .empty;
                for (content.?.array, 0..) |part, k| {
                    if (k > 0) try text.appendSlice(cx.a, "\n\n");
                    if (isImagePart(part)) {
                        try im.list.append(cx.a, try imageBytes(cx, part));
                        try text.appendSlice(cx.a, im.placeholder);
                        continue;
                    }
                    const typed = part == .object and part.get("type") != null and part.get("type").? == .string and std.mem.eql(u8, part.get("type").?.string, "text");
                    if (!typed or fields.hasMedia(part)) return cx.refuse(text_only_refusal);
                    const t = part.get("text") orelse return cx.refuse("image input: a text part needs a text string");
                    if (t != .string) return cx.refuse("image input: a text part needs a text string");
                    try text.appendSlice(cx.a, try escape(cx, t.string, im.placeholder));
                }
                item = try withField(cx, obj, "content", .{ .string = text.items });
            } else if (content == null or content.? == .null) {
                item = try withField(cx, obj, "content", .{ .string = "" });
            } else if (content.? != .string) {
                return cx.refuse("message content must be text or an array of text parts");
            }
            try place(cx, &out, &instructions, item, role.?.string, late_system, dialect);
            continue;
        }
        if (fields.hasMedia(message)) return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
        if (dialect == .deepseek_v41) {
            try place(cx, &out, &instructions, try dsv41Message(cx, message.object, role.?.string), role.?.string, late_system, dialect);
            continue;
        }
        const content = message.get("content");
        var item: *json.Object = message.object;
        if (content != null and content.? == .array) {
            var text: std.ArrayList(u8) = .empty;
            for (content.?.array) |part| {
                const typed = part == .object and part.get("type") != null and part.get("type").? == .string and std.mem.eql(u8, part.get("type").?.string, "text");
                if (!typed or fields.hasMedia(part)) return cx.refuse("this server accepts text parts only; image, audio and video inputs are unsupported");
                const t = part.get("text") orelse return cx.refuse("a text content part must contain a text string");
                if (t != .string) return cx.refuse("a text content part must contain a text string");
                try text.appendSlice(cx.a, t.string);
            }
            item = try withField(cx, message.object, "content", .{ .string = text.items });
        } else if (content == null or content.? == .null) {
            item = try withField(cx, message.object, "content", .{ .string = "" });
        } else if (content.? != .string) {
            return cx.refuse("message content must be text or an array of text parts");
        }
        try place(cx, &out, &instructions, item, role.?.string, late_system, dialect);
    }
    if (instructions.items.len > 0) {
        var joined: std.ArrayList(u8) = .empty;
        for (instructions.items, 0..) |m, i| {
            if (i > 0) try joined.appendSlice(cx.a, "\n\n");
            try joined.appendSlice(cx.a, m.get("content").?.string);
        }
        const first = try withField(cx, instructions.items[0], "role", .{ .string = "system" });
        try first.put(cx.a, "content", .{ .string = joined.items });
        try out.insert(cx.a, 0, .{ .object = first });
    }
    // A template that demands a user query (it raises "No user query found") refuses a conversation whose user turn became tool results; by the template's test a user turn wholly inside a <|im_start|> block is no query, so only a conversation without one gains a placeholder user turn after its last tool run.
    if (needs_user_after_tool) {
        var has_query = false;
        var last_tool: ?usize = null;
        for (out.items, 0..) |m, i| {
            const role = m.get("role");
            if (role == null or role.? != .string) continue;
            if (std.mem.eql(u8, role.?.string, "user")) {
                const content = m.get("content");
                const text = if (content != null and content.? == .string) content.?.string else "";
                const trimmed = std.mem.trim(u8, text, " \t\n\r\x0b\x0c");
                if (!(std.mem.startsWith(u8, trimmed, "<tool_response>") and std.mem.endsWith(u8, trimmed, "</tool_response>"))) has_query = true;
            } else if (std.mem.eql(u8, role.?.string, "tool")) {
                last_tool = i;
            }
        }
        if (!has_query) {
            const at = if (last_tool) |i| i + 1 else out.items.len;
            const turn = try json.newObject(cx.a);
            try turn.put(cx.a, "role", .{ .string = "user" });
            try turn.put(cx.a, "content", .{ .string = "(tool results above)" });
            try out.insert(cx.a, at, .{ .object = turn });
        }
    }
    return .{ .array = out.items };
}

/// A normalized message into the list: leading instructions held to be merged, a later one as ``late_system``. Under
/// DeepSeek-V4.1's rules nothing is merged: its encoder gives every system message its own block, so a leading one
/// stays in place as "system" (the encoder knows no developer role) and the next ones render as mid-conversation
/// system messages, as they do there.
fn place(cx: *Cx, out: *std.ArrayList(Value), instructions: *std.ArrayList(*json.Object), message: *json.Object, r: []const u8, late_system: []const u8, dialect: Dialect) errors.Refused!void {
    var item = message;
    if (std.mem.eql(u8, r, "system") or std.mem.eql(u8, r, "developer")) {
        if (out.items.len == 0 and dialect == .generic) {
            try instructions.append(cx.a, item);
            return;
        }
        const as = if (out.items.len == 0) "system" else late_system;
        if (!std.mem.eql(u8, r, as)) item = try withField(cx, item, "role", .{ .string = as });
    }
    try out.append(cx.a, .{ .object = item });
}

/// ``_normalize_tool_call_arguments``: call arguments as objects for templates; bad ones under ``_invalid_arguments``.
/// DeepSeek-V4.1's encoder decodes a JSON string up to twice (``encode_arguments_to_dsml``), so double-encoded
/// arguments reach their object there.
pub fn toolArguments(cx: *Cx, messages: Value, dialect: Dialect) errors.Refused!Value {
    if (messages != .array) return messages;
    const out = try cx.a.alloc(Value, messages.array.len);
    for (messages.array, out) |message, *slot| {
        slot.* = message;
        const calls = message.get("tool_calls") orelse continue;
        if (calls != .array or calls.array.len == 0) continue;
        var touched = false;
        const new_calls = try cx.a.alloc(Value, calls.array.len);
        for (calls.array, new_calls) |call, *dst| {
            dst.* = call;
            const function = call.get("function") orelse continue;
            if (function != .object or !function.has("arguments")) continue;
            const args = function.get("arguments").?;
            if (args == .object) continue;
            var parsed: ?Value = args;
            if (args == .string) {
                parsed = switch (try json.parseText(cx.a, args.string)) {
                    .ok => |v| v,
                    .err => null,
                };
                if (dialect == .deepseek_v41) if (parsed) |p| if (p == .string) {
                    parsed = switch (try json.parseText(cx.a, p.string)) {
                        .ok => |v| v,
                        .err => p,
                    };
                };
            }
            if (parsed == null or parsed.? != .object) {
                const wrapped = try json.newObject(cx.a);
                try wrapped.put(cx.a, "_invalid_arguments", args);
                parsed = .{ .object = wrapped };
            }
            const fn_copy = try withField(cx, function.object, "arguments", parsed.?);
            dst.* = .{ .object = try withField(cx, call.object, "function", .{ .object = fn_copy }) };
            touched = true;
        }
        if (touched) slot.* = .{ .object = try withField(cx, message.object, "tool_calls", .{ .array = new_calls }) };
    }
    return .{ .array = out };
}

/// ``{"type": "text", "text": text}``: a chat content part.
pub fn textPart(a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error!Value {
    const p = try json.newObject(a);
    try p.put(a, "type", .{ .string = "text" });
    try p.put(a, "text", .{ .string = text });
    return .{ .object = p };
}

/// A short title request without tools (``is_title_request``): it yields to foreground turns.
pub fn isTitleRequest(messages: Value, has_tools: bool) bool {
    if (has_tools or messages != .array or messages.array.len == 0) return false;
    const first = messages.array[0];
    const role = first.get("role") orelse return false;
    if (role != .string or !std.mem.eql(u8, role.string, "system")) return false;
    var size: usize = 0;
    for (messages.array) |m| {
        const c = m.get("content");
        size += if (c != null and c.? == .string) std.unicode.utf8CountCodepoints(c.?.string) catch c.?.string.len else 4096;
    }
    const text = first.get("content") orelse return false;
    if (size >= 4096 or text != .string) return false;
    return std.ascii.findIgnoreCase(text.string, "title") != null;
}

test "a conversation with a user query stays byte-identical under the rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "user", "content": "run it"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "g", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t2", "content": "ok"}, {"role": "assistant", "content": "done"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", true, .generic);
    // byte-identical: the same messages, in order, nothing gained
    try std.testing.expectEqualStrings(try json.stringify(cx.a, messages, .{ .ascii = false }), try json.stringify(cx.a, out, .{ .ascii = false }));
}

test "a conversation with no user query gains one placeholder turn after its last tool run" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}, {"role": "tool", "tool_call_id": "t2", "content": "ok"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", true, .generic);
    try std.testing.expectEqual(@as(usize, 4), out.array.len);
    try std.testing.expectEqualStrings("tool", out.array[1].get("role").?.string);
    try std.testing.expectEqualStrings("42 rows", out.array[1].get("content").?.string); // the tool messages unchanged
    try std.testing.expectEqualStrings("tool", out.array[2].get("role").?.string);
    try std.testing.expectEqualStrings("ok", out.array[2].get("content").?.string);
    try std.testing.expectEqualStrings("user", out.array[3].get("role").?.string);
    try std.testing.expectEqualStrings("(tool results above)", out.array[3].get("content").?.string);
}

test "a user message whose content is wholly a tool block does not count as a query" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const user = try json.newObject(cx.a);
    try user.put(cx.a, "role", .{ .string = "user" });
    try user.put(cx.a, "content", .{ .string = "<tool_response>\ngot 42 rows\n</tool_response>" });
    const assistant = try json.newObject(cx.a);
    try assistant.put(cx.a, "role", .{ .string = "assistant" });
    try assistant.put(cx.a, "content", .{ .string = "" });
    const tool = try json.newObject(cx.a);
    try tool.put(cx.a, "role", .{ .string = "tool" });
    try tool.put(cx.a, "tool_call_id", .{ .string = "t1" });
    try tool.put(cx.a, "content", .{ .string = "done" });
    const messages: Value = .{ .array = try cx.a.dupe(Value, &.{ .{ .object = user }, .{ .object = assistant }, .{ .object = tool } }) };
    const out = try normalize(&cx, messages, "system", true, .generic);
    try std.testing.expectEqual(@as(usize, 4), out.array.len);
    try std.testing.expectEqualStrings("user", out.array[3].get("role").?.string);
    try std.testing.expectEqualStrings("(tool results above)", out.array[3].get("content").?.string);
}

test "a conversation keeps its shape when the template does not need the rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", false, .generic);
    try std.testing.expectEqual(@as(usize, 2), out.array.len);
    try std.testing.expectEqualStrings("assistant", out.array[0].get("role").?.string);
    try std.testing.expectEqualStrings("tool", out.array[1].get("role").?.string);
}

test "a message's own text holding the image placeholder is escaped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const ph = "<\u{ff5c}deepseek_image\u{ff5c}>";
    var images: Images = .{ .placeholder = ph, .max = 32 };
    const messages = (try json.parseText(cx.a,
        "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"see <\u{ff5c}deepseek_image\u{ff5c}> above\"}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,aGVsbG8=\"}}]}]"
    )).ok;
    const out = try normalizeImages(&cx, messages, "system", false, .generic, &images);
    const text = out.array[0].get("content").?.string;
    // the image part's own placeholder stays; the text's is escaped to plain ASCII bars
    try std.testing.expectEqualStrings("see <|deepseek_image|> above\n\n" ++ ph, text);
    try std.testing.expectEqual(@as(usize, 1), images.list.items.len);
}

// DeepSeek-V4.1 dialect: regression cases from w-ref's chat-template parity corpus (node0 ~/tfp/ref/chat, b841319),
// each the input of a named case that diverged from the official encoder on strix-port 352d274.

fn dsv41Normalize(cx: *Cx, text: []const u8) errors.Refused!Value {
    return normalize(cx, (try json.parseText(cx.a, text)).ok, "system", false, .deepseek_v41);
}

test "dialectOf: DeepSeek-V4.1's latest_reminder token and DSML invoke tags" {
    try std.testing.expectEqual(Dialect.deepseek_v41, dialectOf("{%- set REMINDER = '<\u{ff5c}latest_reminder\u{ff5c}>' -%} <\u{ff5c}DSML\u{ff5c} invoke name=\"$TOOL_NAME\">"));
    try std.testing.expectEqual(Dialect.generic, dialectOf("{%- for message in messages %}<|im_start|>{{ message.role }}{% endfor %}"));
    try std.testing.expectEqual(Dialect.generic, dialectOf("<\u{ff5c}latest_reminder\u{ff5c}> without the DSML tags"));
}

test "U33 content parts joined by a blank line (corpus: user content parts; system and assistant content parts; tool result content parts)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const user = "[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"one\"}, {\"type\": \"text\", \"text\": \"two\"}]}]";
    try std.testing.expectEqualStrings("one\n\ntwo", (try dsv41Normalize(&cx, user)).array[1].get("content").?.string);
    // the generic rules stay as they were: concatenated
    try std.testing.expectEqualStrings("onetwo", (try normalize(&cx, (try json.parseText(cx.a, user)).ok, "system", false, .generic)).array[1].get("content").?.string);
    const sys_asst = (try dsv41Normalize(&cx, "[{\"role\": \"system\", \"content\": [{\"type\": \"text\", \"text\": \"s1\"}, {\"type\": \"text\", \"text\": \"s2\"}]}, {\"role\": \"user\", \"content\": \"Hello\"}, {\"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"a1\"}, {\"type\": \"text\", \"text\": \"a2\"}]}, {\"role\": \"user\", \"content\": \"q\"}]")).array;
    try std.testing.expectEqualStrings("s1\n\ns2", sys_asst[0].get("content").?.string);
    try std.testing.expectEqualStrings("a1\n\na2", sys_asst[2].get("content").?.string);
    const tool = (try dsv41Normalize(&cx, "[{\"role\": \"user\", \"content\": \"q\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"c1\", \"type\": \"function\", \"function\": {\"name\": \"a\", \"arguments\": {}}}]}, {\"role\": \"tool\", \"tool_call_id\": \"c1\", \"content\": [{\"type\": \"text\", \"text\": \"t1\"}, {\"type\": \"text\", \"text\": \"t2\"}]}]")).array;
    try std.testing.expectEqualStrings("t1\n\nt2", tool[2].get("content").?.string);
    // the encoder's join keeps text parts only: another part type of a system or assistant message is dropped
    const dropped = (try dsv41Normalize(&cx, "[{\"role\": \"user\", \"content\": \"q\"}, {\"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"a\"}, {\"type\": \"refusal\", \"refusal\": \"no\"}]}]")).array;
    try std.testing.expectEqualStrings("a", dropped[1].get("content").?.string);
}

test "U34 the image placeholder in text is refused (corpus: literal image placeholder in user text; in reasoning_content)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const cases = [_][]const u8{
        "[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"see <\u{ff5c}deepseek_image\u{ff5c}> here\"}]",
        "[{\"role\": \"user\", \"content\": \"Hello\"}, {\"role\": \"assistant\", \"content\": \"a\", \"reasoning_content\": \"r <\u{ff5c}deepseek_image\u{ff5c}>\"}, {\"role\": \"user\", \"content\": \"q\"}]",
        "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"x <\u{ff5c}deepseek_image\u{ff5c}>\"}]}]",
        "[{\"role\": \"user\", \"content\": [{\"type\": \"tool_result\", \"tool_use_id\": \"c1\", \"content\": [{\"type\": \"text\", \"text\": \"<\u{ff5c}deepseek_image\u{ff5c}>\"}]}]}]",
    };
    for (cases) |c| {
        try std.testing.expectError(error.Refused, dsv41Normalize(&cx, c));
        try std.testing.expectEqualStrings(deepseek_image_refusal, cx.message);
    }
    // a tool_result part's string content is not checked, as the encoder does not check it
    _ = try dsv41Normalize(&cx, "[{\"role\": \"user\", \"content\": [{\"type\": \"tool_result\", \"tool_use_id\": \"c1\", \"content\": \"<\u{ff5c}deepseek_image\u{ff5c}>\"}]}]");
    // image parts stay refused on a text-only request
    try std.testing.expectError(error.Refused, dsv41Normalize(&cx, "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"look\"}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"x.png\"}}]}]"));
}

test "U35 latest_reminder, an empty conversation and tool_result parts are taken (corpus: latest_reminder; no messages; tool_result parts)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const reminder = (try dsv41Normalize(&cx, "[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"latest_reminder\", \"content\": \"2026-02-21,Saturday\"}, {\"role\": \"user\", \"content\": \"Hello\"}]")).array;
    try std.testing.expectEqual(@as(usize, 3), reminder.len);
    try std.testing.expectEqualStrings("latest_reminder", reminder[1].get("role").?.string);
    try std.testing.expectEqual(@as(usize, 0), (try dsv41Normalize(&cx, "[]")).array.len);
    try std.testing.expectError(error.Refused, normalize(&cx, (try json.parseText(cx.a, "[]")).ok, "system", false, .generic));
    // a user message of tool results and text keeps its parts: the template renders and orders the blocks
    const parts = (try dsv41Normalize(&cx, "[{\"role\": \"user\", \"content\": \"q\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"c1\", \"type\": \"function\", \"function\": {\"name\": \"a\", \"arguments\": {}}}, {\"id\": \"c2\", \"type\": \"function\", \"function\": {\"name\": \"b\", \"arguments\": {}}}]}, {\"role\": \"user\", \"content\": [{\"type\": \"tool_result\", \"tool_use_id\": \"c2\", \"content\": \"two\"}, {\"type\": \"text\", \"text\": \"and\"}, {\"type\": \"tool_result\", \"tool_use_id\": \"c1\", \"content\": \"one\"}]}]")).array;
    const kept = parts[2].get("content").?;
    try std.testing.expect(kept == .array);
    try std.testing.expectEqual(@as(usize, 3), kept.array.len);
    try std.testing.expectEqualStrings("tool_result", kept.array[0].get("type").?.string);
}

test "U36 double-encoded call arguments decoded twice (corpus: tool call arguments double-encoded JSON)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const msgs = (try json.parseText(cx.a, "[{\"role\": \"user\", \"content\": \"q\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"c1\", \"type\": \"function\", \"function\": {\"name\": \"a\", \"arguments\": \"\\\"{\\\\\\\"x\\\\\\\": 1}\\\"\"}}]}]")).ok;
    const v41 = (try toolArguments(&cx, msgs, .deepseek_v41)).array[1].get("tool_calls").?.array[0].get("function").?.get("arguments").?;
    try std.testing.expect(v41 == .object);
    try std.testing.expectEqualStrings("1", v41.get("x").?.int);
    // the generic rule decodes once and keeps the inner string as invalid arguments
    const generic = (try toolArguments(&cx, msgs, .generic)).array[1].get("tool_calls").?.array[0].get("function").?.get("arguments").?;
    try std.testing.expect(generic.get("_invalid_arguments") != null);
    // a string that does not decode to an object stays the original value, as the encoder's {"arguments": ...}
    const odd = (try json.parseText(cx.a, "[{\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"type\": \"function\", \"function\": {\"name\": \"a\", \"arguments\": \"\\\"plain\\\"\"}}]}]")).ok;
    const kept = (try toolArguments(&cx, odd, .deepseek_v41)).array[0].get("tool_calls").?.array[0].get("function").?.get("arguments").?;
    try std.testing.expectEqualStrings("\"plain\"", kept.get("_invalid_arguments").?.string);
}

test "U38 leading system messages each keep their own block under DeepSeek-V4.1's rules, merged under the generic ones" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const text = "[{\"role\": \"system\", \"content\": \"A\"}, {\"role\": \"developer\", \"content\": \"B\"}, {\"role\": \"system\", \"content\": \"C\"}, {\"role\": \"user\", \"content\": \"u\"}]";
    const v41 = (try dsv41Normalize(&cx, text)).array;
    try std.testing.expectEqual(@as(usize, 4), v41.len);
    for ([_][]const u8{ "A", "B", "C" }, 0..) |want, k| {
        try std.testing.expectEqualStrings("system", v41[k].get("role").?.string); // the encoder knows no developer role
        try std.testing.expectEqualStrings(want, v41[k].get("content").?.string);
    }
    const generic = (try normalize(&cx, (try json.parseText(cx.a, text)).ok, "system", false, .generic)).array;
    try std.testing.expectEqual(@as(usize, 2), generic.len);
    try std.testing.expectEqualStrings("A\n\nB\n\nC", generic[0].get("content").?.string);
}

test "U39 null and absent content as DeepSeek-V4.1's encoder writes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    // a user message's null content: the encoder's join fails, so the request is refused
    try std.testing.expectError(error.Refused, dsv41Normalize(&cx, "[{\"role\": \"user\", \"content\": null}]"));
    const out = (try dsv41Normalize(&cx,
        "[{\"role\": \"system\", \"content\": null}, {\"role\": \"latest_reminder\"}, {\"role\": \"user\"}, {\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"id\": \"c1\", \"type\": \"function\", \"function\": {\"name\": \"a\", \"arguments\": {}}}]}, {\"role\": \"tool\", \"tool_call_id\": \"c1\", \"content\": null}, {\"role\": \"tool\", \"tool_call_id\": \"c2\"}, {\"role\": \"latest_reminder\", \"content\": null}, {\"role\": \"user\", \"content\": [{\"type\": \"tool_result\", \"tool_use_id\": \"c1\", \"content\": null}, {\"type\": \"tool_result\", \"tool_use_id\": \"c2\"}]}]",
    )).array;
    const want = [_][]const u8{ "", "None", "", "", "None", "", "None" };
    for (want, 0..) |w, k| try std.testing.expectEqualStrings(w, out[k].get("content").?.string);
    // a tool_result part's null content is the text None; an absent one stays absent (the template writes nothing)
    const parts = out[7].get("content").?.array;
    try std.testing.expectEqualStrings("None", parts[0].get("content").?.string);
    try std.testing.expect(parts[1].get("content") == null);
}
