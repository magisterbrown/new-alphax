python3 test_bot.py &> btest.log &
while ! madbg connect 127.0.0.1 3513; do sleep 0.5; done
