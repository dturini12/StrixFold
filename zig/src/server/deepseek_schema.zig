//! A request's response_format as the schema DeepSeek-V4.1's encoder shows under ``## Response Format``.
const std = @import("std");
const json = @import("json.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// ``schema_for_prompt``: what ``## Response Format`` shows for a ``response_format``, or null (text, absent or not
/// a shape that names a schema). json_object shows ``{"type": "object"}``; json_schema its schema (a JSON string of one
/// parsed).
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
    return schema;
}
