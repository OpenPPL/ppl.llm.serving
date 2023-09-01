for num in {1..300}
do
    ./client_pressure host:port &
    usleep 50000
done
wait