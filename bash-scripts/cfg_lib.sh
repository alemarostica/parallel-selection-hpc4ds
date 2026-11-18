# bench "<label>" <selection options...> [k]
# runs mom, rand, fr. 3 repetitions each, with parseable header per run
RANKS=${RANKS:-16}
bench(){
    local label="$1"; shift
    for p in mom rand fr; do
        for rep in 1 2 3; do
            echo "### $label | pivot=$p | rep=$rep"
            mpirun -n $RANKS ./mpi_wrapper.sh ./selection -p $p -d -V "$@"
        done
    done
}
