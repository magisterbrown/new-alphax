pgrep uwsgi | xargs kill &> /dev/null;
pgrep madbg | xargs kill &> /dev/null;
redis-cli LTRIM to_analyze 1 0
sleep 0.4
uwsgi --http :9000 --wsgi-file server.py
