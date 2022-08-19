This is a bare-bone implementation of HELICS driven OPF simulation 
for IEEE 123 Bus PV system. 

To run the program:

helics run --path=feeder_debugger_runner.json

Two Federates exist:
- OPF Federate 
  - files are placed in "opf_federate_123PV" directory
- DSS Federate 
  - Few changes from the original SGIDAL Working Example's Feeder Federate has been done
  - files are place in "dss_federate_123PV" directory and the specific changes are mentioned in the README.md of that folder

Dependency:
- "pip install -r requirements.txt" will give you all of the packages except "ipopt"
IPOPT:
- IPOPT installation changes based on Windows/Linux/Mac if you want to install with pip
- If you use miniconda, then it is simple and can be installed using conda-forge

Files Structure:
- clean.sh: script to clean log files
- kill.sh: script to kill helics federate manually 
- feeder_opf_runner.json: script to run opf script
- Directory "dss_federate_123PV"
- Directory "opf_federate_123PV"