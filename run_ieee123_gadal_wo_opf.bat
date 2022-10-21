start /b cmd /c python test_full_systems_ieee123_gadal.py test_ieee123_gadal.json ^> setup_log.log 2^>^&1
start /b cmd /c helics run --path=test_system_runner_ieee123_gadal.json ^> runner_log.log 2^>^&1
