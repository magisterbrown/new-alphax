pgrep uwsgi | xargs kill -9 &> /dev/null;
pgrep madbg | xargs kill -9 &> /dev/null;
rm -r board_logs/*
redis-cli LTRIM to_analyze 1 0
redis-cli LTRIM to_learn 1 0
sleep 0.4

