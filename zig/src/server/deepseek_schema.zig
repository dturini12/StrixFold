//! A request's response_format as the schema DeepSeek-V4.1's encoder shows under ``## Response Format``.
const std = @import("std");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const template = @import("template");
const toStd = @import("hf_text.zig").toStd;
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// ``schema_for_prompt``: json_object as ``{"type": "object"}``, json_schema as its (parsed) schema; falsy: null.
pub fn schemaFor(a: Allocator, rf: Value) Allocator.Error!?Value {
    if (rf != .object) return null;
    const kind = rf.get("type") orelse return null;
    if (kind != .string) return null;
    if (std.mem.eql(u8, kind.string, "json_object")) return switch (try json.parseText(a, "{\"type\": \"object\"}")) {
        .ok => |v| v,
        .err => null,
    };
    if (!std.mem.eql(u8, kind.string, "json_schema")) return null;
    const js = rf.get("json_schema") orelse return null;
    if (js != .object) return null;
    var schema = js.get("schema") orelse js.get("json_schema") orelse return null;
    if (schema == .string) schema = switch (try json.parseText(a, schema.string)) {
        .ok => |v| v,
        .err => return null,
    };
    return if (schema.truthy()) schema else null; // the encoder's ``if response_format``: {} shows no block
}

/// The block encoding.py ends a system message with; tojson as the chat template's (the same serializer).
const block_source = "{{- '\\n\\n## Response Format:\\n\\nYou MUST strictly adhere to the following schema to reply:\\n' ~ (response_format | tojson) -}}";

/// The chat template's text with `schema`'s block closing the first system message (an empty one added if none).
pub fn render(text: model_text.Text, a: Allocator, messages: Value, options: model_text.RenderOptions, schema: Value, problem: *[]const u8) model_text.Error![]u8 {
    const msgs = try withSystem(a, messages);
    const full = try text.render(a, msgs, options, problem);
    var solo = options;
    solo.add_generation_prompt = false;
    const upto = try text.render(a, .{ .array = msgs.array[0..1] }, solo, problem);
    if (!std.mem.startsWith(u8, full, upto)) {
        problem.* = "the chat template's first system message is not a prefix of its prompt: no place for the response_format block";
        return error.Template;
    }
    var context: std.json.ObjectMap = .empty;
    try context.put(a, "response_format", try toStd(a, schema));
    var diag: template.Diag = .{};
    const block = template.renderChat(a, block_source, .null, .null, .{ .object = context }, false, &diag) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        problem.* = diag.msg;
        return error.Template;
    };
    return std.mem.concat(a, u8, &.{ upto, block, full[upto.len..] });
}

/// Text.renderIds with render's block: the text, then its ids.
pub fn renderIds(text: model_text.Text, a: Allocator, messages: Value, options: model_text.RenderOptions, schema: Value, problem: *[]const u8) model_text.Error![]u32 {
    return text.encode(a, try render(text, a, messages, options, schema, problem), false);
}

/// `messages` led by a system message: an empty one first unless the first is a system or developer message.
fn withSystem(a: Allocator, messages: Value) Allocator.Error!Value {
    const list: []const Value = if (messages == .array) messages.array else &.{};
    if (list.len > 0) if (list[0].strField("role")) |role| {
        if (std.mem.eql(u8, role, "system") or std.mem.eql(u8, role, "developer")) return .{ .array = @constCast(list) };
    };
    const system = try json.newObject(a);
    try system.put(a, "role", .{ .string = "system" });
    try system.put(a, "content", .{ .string = "" });
    const out = try a.alloc(Value, list.len + 1);
    out[0] = .{ .object = system };
    @memcpy(out[1..], list);
    return .{ .array = out };
}
