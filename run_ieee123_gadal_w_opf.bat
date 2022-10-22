start /b cmd /c python test_full_systems_ieee123_gadal_w_opf.py test_ieee123_gadal_w_opf.json ^> setup_log.log 2^>^&1
start /b cmd /c helics run --path=test_system_runner_ieee123_gadal_w_opf.json ^> runner_log.log 2^>^&1
