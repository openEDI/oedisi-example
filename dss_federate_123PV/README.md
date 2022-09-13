This is the DSS Federate directory
- Mostly, it uses the SGIDAL working example dss federate, 
however some of the new scripts written for OPF are below and appropriate functions are added in
sender_cosim.py and FeederSimulator.py

- Files Structure:
- OPENDSS files for generating adjacency matrices: 
  - ieee123pv_EXP_Y, ieee123pv_Inc_Matrix.csv, ieee123pv_Inc_Matrix_Cols.csv, ieee123pv_Inc_Matrix_Rows.csv
- /IEEE123PV: feeder model for IEEE 123 PV
- opf_dss_functions.py: supporting scripts for generating vectors, matrices to be utilized in opf
- opf_incidence_matrices.py: "incidence"/"adjacency" matrices generation script to identify controllable variables with respect to Y bus node order
- feeder_input_mapping.json: subscribed messages of controllable variables from opf federate
- feeder_static_inputs_ieee123PV.json: configuration files 
- /base_vals: safe-fail variables if OPF fails