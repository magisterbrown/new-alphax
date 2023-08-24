pgrep uwsgi | xargs kill -9 &> /dev/null;
pgrep madbg | xargs kill -9 &> /dev/null;
redis-cli LTRIM to_analyze 1 0
redis-cli LTRIM to_learn 1 0
sleep 0.4
uwsgi --http :9000 --wsgi-file server.py &> wsgi.out &
#python3 test_server.py &> tests.log &
while ! madbg connect 127.0.0.1 3513; do sleep 0.5; done

