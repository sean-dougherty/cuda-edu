function err() {
    echo
    echo "$@" 1>&2
    exit 1
}

function get_pfm() {
    local un=$(uname)
    if echo $un | grep -i cygwin &> /dev/null; then
        echo "Cygwin"
    else
	    echo $un
    fi
}

function find_program() {
    local description="$1"
    local candidates="$2"
    local result=""

    printf "Looking for ${description}..." > /dev/tty

    for x in ${candidates}; do
        if which $x &>/dev/null; then
            result=$(which $x)
            break
        fi
    done

    if [ -z "${result}" ]; then
        err "Cannot locate ${description}."
    fi
    printf "OK: ${result}\n" > /dev/tty

    echo "${result}"
}

function find_file() {
    local description="$1"
    local subpath="$2"
    local sane_location="$3"
    local find_root="$4"
    local result=""

    printf "Searching for ${description}... " > /dev/tty
    if [ ! -z "${sane_location}" ] && [ -f "${sane_location}/${subpath}" ]; then
        result="${sane_location}/${subpath}"
    else
        # First try locate, since it's faster.
        result=$(locate "${find_root}/*/${subpath}" | tail -1)
        if [ -z "${result}" ]; then
            # Maybe the locate database isn't complete... fall back to find.
            result=$(find -L "${find_root}" -path "*/${subpath}" | tail -1)
        fi
    fi

    if [ -z "${result}" ]; then
        err "Cannot locate ${find_path}"
    fi
    printf "OK: ${result}\n" > /dev/tty
    
    echo "${result}"
}
