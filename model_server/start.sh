#pgrep uwsgi | xargs kill -9 &> /dev/null;
#pgrep madbg | xargs kill -9 &> /dev/null;
#rm -r run_logs/*
#redis-cli LTRIM to_analyze 1 0
#redis-cli LTRIM to_learn 1 0
#sleep 0.4
./prestart.sh
uwsgi --processes 4 --threads 2 --http :9000 --wsgi-file server.py
