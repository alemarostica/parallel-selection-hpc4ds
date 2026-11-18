# bench "<label>" <selection options...> [k]
# runs mom, rand, fr. 3 repetitions each, with parseable header per run
RANKS=${RANKS:-16}
bench(){
    local label="$1"; shift
    for mode in mom rand fr frR; do
        local p=${mode%R}
        local extra=""
        [[ $mode == frR ]] && extra="-R"
        for rep in 1 2 3; do
            echo "### $label | pivot=$mode | rep=$rep"
            mpirun -n $RANKS ./mpi_wrapper.sh ./selection -p $p $extra -d -V "$@"
        done
    done
}
