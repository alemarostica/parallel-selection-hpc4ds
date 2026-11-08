--[[
    gen_dataset.lua - generates a binary file of random integers.

    Requires lua 5.3+ (native 64-bit integers, bit ops and string.pack).

    Usage: lua gen_dataset.lua <n> <output_file> [seed] [max_value]
      n          number of integers
      seed       RNG seed (default 42)
      max_value  values are uniform in [0, max_value) (default 2^31-1)
]]

local CHUNK = 4096 -- values packed per write

-- splitmix64 PRNG, for the love of the game
local state
local function next_rand()
    state = state + 0x9E3779B97F4A7C15
    local z = state
    z = (z ~ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ~ (z >> 27)) * 0x94D049BB133111EB
    return z ~ (z >> 31)
end

-- Unsigned 64-bit remainder. m must be > 0
local function umod(a, m)
    if a >= 0 then return a % m end
    local q = ((a >> 1) // m) << 1
    local r = a - q * m
    if not math.ult(r, m) then r = r - m end
    return r
end

local function fail(msg)
    io.stderr:write(msg, "\n")
    os.exit(1)
end

if #arg < 2 then
    fail("Usage: lua " .. (arg[0] or "gen_dataset.lua") ..
        " <n> <output_file> [seed] [max_value]")
end

local n = math.tointeger(tonumber(arg[1]))
local path = arg[2]
state = math.tointeger(tonumber(arg[3] or "42"))
local maxv = math.tointeger(tonumber(arg[4] or "2147483647"))

if not (n and state and maxv) or n <= 0 or maxv <= 0 or maxv > 2147483647 then
    fail("Invalid arguments (n > 0, 0 < max_value <= 2^31, integers only)")
end

local f, err = io.open(path, "wb")
if not f then fail("cannot open output file: " .. err) end

f:write(string.pack("<i8", n))

local full_fmt = "<" .. string.rep("i4", CHUNK)
local buf = {}
local remaining = n
while remaining > 0 do
    local m = math.min(CHUNK, remaining)
    for i = 1, m do
        buf[i] = umod(next_rand(), maxv)
    end
    local fmt = (m == CHUNK) and full_fmt or ("<" .. string.rep("i4", m))
    f:write(string.pack(fmt, table.unpack(buf, 1, m)))
    remaining = remaining - m
end

f:close()
print(string.format("Wrote %d integers to %s (%.2f MB)",
                    n, path, (8 + 4 * n) / (1024 * 1024)))


