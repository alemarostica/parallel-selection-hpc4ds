# gen_dataset.tcl - generates a binary file of random integers
set CHUNK 4096
set MASK 0xFFFFFFFFFFFFFFFF

# splitmix64
proc next_rand {} {
    global state MASK
    set state [expr {($state + 0x9E3779B97F4A7C15) & $MASK}]
    set z $state
    set z [expr {(($z ^ ($z >> 30)) * 0xBF58476D1CE4E5B9) & $MASK}]
    set z [expr {(($z ^ ($z >> 27)) * 0x94D049BB133111EB) & $MASK}]
    return [expr {$z ^ ($z >> 31)}]
}

proc fail {msg} {
    puts stderr $msg
    exit 1
}

if {$argc < 2} {
    fail "Usage: tclsh $argv0 <n> <output_file> \[seed\] \[max_value\]"
}

set n [lindex $argv 0]
set path [lindex $argv 1]
set state [expr {$argc > 2 ? [lindex $argv 2] : 42}]
set maxv [expr {$argc > 3 ? [lindex $argv 3] : 2147483647}]

foreach v [list $n $state $maxv] {
    if {![string is wideinteger -strict $v]} {
        fail "Invalid arguments (integers only)"
    }
}
if {$n <= 0 || $maxv <= 0 || $maxv > 2147483648} {
    fail "Invalid arguments (n > 0, 0 < max_value <= 2^31)"
}
set state [expr {$state & $MASK}]

if {[catch {open $path wb} f]} {
    fail "cannot open output file: $f"
}
fconfigure $f -translation binary

puts -nonewline $f [binary format w $n]

set remaining $n
while {$remaining > 0} {
    set m [expr {min($remaining, $CHUNK)}]
    set vals {}
    for {set i 0} {$i < $m} {incr i} {
        lappend vals [expr {[next_rand] % $maxv}]
    }
    puts -nonewline $f [binary format i* $vals]
    incr remaining -$m
}

close $f
puts [format "Wrote %d integers to %s (%.2f MB)" $n $path \
    [expr {(8 + 4.0 * $n) / (1024 * 1024)}]]
