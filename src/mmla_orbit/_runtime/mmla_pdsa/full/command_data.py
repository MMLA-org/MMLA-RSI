"""Lossless JSON data quoting for emitted commands, never action repair."""
import json
POLICY='orbit-command-control-json-unicode-v1'

def escape_control_json(wire,controls):
    """Escape literal control spellings while preserving the entire JSON value.

    Longer matches go first, including slash controls and partial token prefixes.
    JSON is already serialized, so escaped quotes/backslashes cannot be confused
    with payload delimiters. Pinned controls start with ASCII '<' or '/'.
    """
    if type(wire) is not str:raise TypeError('serialized JSON string required')
    matches=tuple(sorted(set(controls),key=lambda t:(-len(t),t)))
    if any(type(t) is not str or not t or t[0] not in ('<','/') for t in matches):
        raise ValueError('reviewed control spellings must start with less-than or slash')
    result=wire
    for token in matches:
        result=result.replace(token,'\\u%04x'%ord(token[0])+token[1:])
    if any(token in result for token in matches) or json.loads(result)!=json.loads(wire):
        raise ValueError('lossless complete JSON control quoting required')
    return result
