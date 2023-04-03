import logging
import math

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def get_vnom2(dss):
    circuit = dss.Circuit
    AllNodeNames = circuit.AllNodeNames()
    # This or DSSCircuit.AllBusVmagPu
    Vnom = circuit.AllBusMagPu()  # circuit.AllBusVmagPu()

    vmags = circuit.AllBusVMag()
    Vnom2 = circuit.AllBusVolts()
    test_Vnom2 = np.array(
        [complex(Vnom2[i], Vnom2[i + 1]) for i in range(0, len(Vnom2), 2)]
    )
    test_Vnom2_dict = {
        AllNodeNames[ii].upper(): test_Vnom2[ii] for ii in range(len(test_Vnom2))
    }

    test_vmag_volts_result = np.allclose(vmags, np.abs(test_Vnom2))
    logger.debug("test_vmag_volts_result", test_vmag_volts_result)

    AllNodeNamesY = circuit.YNodeOrder()
    yv = circuit.YNodeVArray()
    test_yv = np.array([complex(yv[i], yv[i + 1]) for i in range(0, len(yv), 2)])
    test_yv_dict = {AllNodeNamesY[ii]: test_yv[ii] for ii in range(len(test_yv))}
    test_yv_result = np.allclose(vmags, np.abs(test_yv))
    logger.debug("test_yv_result", test_yv_result)

    logger.debug("Test dictionary")
    for i in test_yv_dict.keys():
        if abs(abs(test_Vnom2_dict[i]) - abs(test_yv_dict[i])) > 0.0001:
            logger.debug(i, abs(test_Vnom2_dict[i]), abs(test_yv_dict[i]))
    # for t1, t2 in zip(np.abs(test_Vnom2), np.abs(test_yv)):
    # np.testing.assert_array_almost_equal(np.abs(test_Vnom2), np.abs(test_yv),decimal=5)

    V = np.ones(len(Vnom) // 2, dtype=np.complex_)
    for i in range(len(V)):
        V[i] = complex(Vnom[2 * i], Vnom[2 * i + 1])
    vnom_dict = {AllNodeNames[ii].upper(): V[ii] for ii in range(len(V))}
    return V, vnom_dict


def get_vnom(dss):
    dss.run_command("BatchEdit Load..* enabled=no")
    dss.run_command("BatchEdit Generator..* enabled=no")
    dss.run_command("solve mode=snap")

    circuit = dss.Circuit
    AllNodeNames = circuit.AllNodeNames()
    Vnom = circuit.AllBusVolts()
    bases = dss.Settings.VoltageBases()
    base = bases[-1] / math.sqrt(3)
    V = np.ones(len(Vnom) // 2, dtype=np.complex_)
    for i in range(len(V)):
        V[i] = complex(Vnom[2 * i], Vnom[2 * i + 1]) / (base * 1000)

    vnom_dict = {AllNodeNames[ii].upper(): V[ii] for ii in range(len(V))}
    dss.run_command("BatchEdit Load..* enabled=yes")
    dss.run_command("BatchEdit Generator..* enabled=yes")
    dss.run_command("solve mode=snap")
    return V, vnom_dict


def get_loads(dss, circuit):
    data = []
    load_flag = dss.Loads.First()
    dss.Circuit.SetActiveClass("Load")
    while load_flag:
        load = dss.Loads
        datum = {
            "name": load.Name(),
            "kV": load.kV(),
            "kW": load.kW(),
            "PF": load.PF(),
            "Delta_conn": load.IsDelta(),
        }
        _ = circuit.SetActiveElement("Load.%s" % datum["name"])
        cktElement = dss.CktElement
        bus = cktElement.BusNames()[0].split(".")
        datum["kVar"] = (
            float(datum["kW"])
            / float(datum["PF"])
            * math.sqrt(1 - float(datum["PF"]) * float(datum["PF"]))
        )
        datum["bus1"] = bus[0]
        datum["numPhases"] = len(bus[1:])
        datum["phases"] = bus[1:]
        if not datum["numPhases"]:
            datum["numPhases"] = 3
            datum["phases"] = ["1", "2", "3"]
        datum["voltageMag"] = cktElement.VoltagesMagAng()[0]
        datum["voltageAng"] = cktElement.VoltagesMagAng()[1]
        datum["power"] = dss.CktElement.Powers()[:2]

        data.append(datum)
        load_flag = dss.Loads.Next()

    return data


def get_pvSystems(dss):
    data = []
    PV_flag = dss.PVsystems.First()
    while PV_flag:
        datum = {}
        # PVname = dss.CktElement.Name()
        PVname = dss.PVsystems.Name()
        PVpmpp = dss.PVsystems.Pmpp()
        PVkW = dss.PVsystems.kW()
        PVpf = dss.PVsystems.pf()
        PVkVARated = dss.PVsystems.kVARated()
        PVkvar = dss.PVsystems.kvar()

        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        # PVkV = dss.run_command('? ' + PVname + '.kV')
        # Not included in PVsystems commands for some reason

        datum["name"] = PVname
        datum["bus"] = bus
        datum["phases"] = bus[1:]
        datum["Pmpp"] = PVpmpp
        datum["pf"] = PVpf
        datum["kW"] = PVkW
        datum["kVar"] = PVkvar
        datum["kVARated"] = PVkVARated
        datum["numPhase"] = NumPhase
        datum["numPhases"] = NumPhase
        datum["power"] = dss.CktElement.Powers()[: 2 * NumPhase]

        data.append(datum)
        PV_flag = dss.PVsystems.Next()
    return data


def get_Generator(dss):
    data = []
    gen_flag = dss.Generators.First()
    dss.Circuit.SetActiveClass("Generator")
    while gen_flag:
        GENname = dss.Generators.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        GENkW = dss.Generators.kW()
        GENpf = dss.Generators.PF()
        GENkV = dss.Generators.kV()
        bus = bus.split(".")
        if len(bus) == 1:
            bus = bus + ["1", "2", "3"]
        datum = {
            "name": GENname,
            "bus": bus,
            "phases": bus[1:],
            "name_bus": GENname + "." + bus[0],
            "kW": GENkW,
            "kVar": dss.Generators.kvar(),
            "pf": GENpf,
            "kV": GENkV,
            "numPhase": NumPhase,
            "numPhases": NumPhase,
        }
        data.append(datum)
        gen_flag = dss.Generators.Next()
    return data


def get_capacitors(dss):
    data = []
    cap_flag = dss.Capacitors.First()
    dss.Circuit.SetActiveClass("Capacitor")
    while cap_flag:
        datum = {}
        capname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        kvar = dss.Capacitors.kvar()
        datum["name"] = capname
        temp = bus.split(".")
        datum["busname"] = temp[0]
        datum["busphase"] = temp[1:]
        if not datum["busphase"]:
            datum["busphase"] = ["1", "2", "3"]
        datum["kVar"] = kvar
        datum["numPhases"] = NumPhase
        datum["power"] = dss.CktElement.Powers()[: 2 * NumPhase]
        data.append(datum)
        cap_flag = dss.Capacitors.Next()
    return data


def get_voltages(circuit):
    temp_Vbus = circuit.YNodeVArray()
    AllNodeNames = circuit.YNodeOrder()
    node_number = len(AllNodeNames)
    name_voltage_dict = {
        AllNodeNames[ii]: complex(temp_Vbus[ii * 2], temp_Vbus[ii * 2 + 1])
        for ii in range(node_number)
    }
    feeder_voltages = np.array(
        [complex(temp_Vbus[ii * 2], temp_Vbus[ii * 2 + 1]) for ii in range(node_number)]
    )
    # voltage_pu = list(map(lambda x: abs(x[0]) / x[1], zip(feeder_voltages, BASEKV)))
    # print(feeder_voltages)
    # print(BASEKV)
    return feeder_voltages, name_voltage_dict


def get_load_sizes(dss, loads):
    load_size_dict = {}
    for load in loads:
        bus_phases = load["bus1"] + "." + ".".join(load["phases"])
        load_size_dict[load["name"]] = {
            "numPhases": load["numPhases"],
            "bus": load["bus1"],
            "bus_phases": bus_phases,
            "phases": load["phases"],
            "kVar": load["kVar"],
            "kV": load["kV"],
            "kW": load["kW"],
        }
    return load_size_dict
