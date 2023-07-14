import numpy as np
import pandas as pd

def read_flex(state, sector, year):
    fname = 'data/EFSFlexLoadProfiles_High.csv'
    df = pd.read_csv(fname)
    #print(df.columns)
    #print(df.Electrification.unique())
    #print(df.TechnologyAdvancement.unique())
    #print(df.Flexibility.unique())
    #print(df.Year.unique())
    #print(df.State.unique())
    #print(df.Sector.unique())
    #print(df[df.Sector == 'Transportation'].LoadMW.unique())

    tech = 'Moderate'

    flex_b = 'Base'
    flex_e = 'Enhanced'

    #sector = 'Residential'
    #sector = 'Commercial'
    sector_ev = 'Transportation'

    #state = 'CA'  # total MWh = 98761848, commercial = 217342397, 2020 res = 98973985, comm = 153859548
    #state = 'VT'  # total MWh = 2592687
    #state = 'AZ'  # total MWh = 29540735
    #state = 'IA'  # total MWh = 17938337

    energy_b = np.sum(df[(df.Year == year) & (df.State == state) & (df.Sector == sector) & (df.TechnologyAdvancement == tech)
            & (df.Flexibility == flex_b)].LoadMW)
    energy_e = np.sum(df[(df.Year == year) & (df.State == state) & (df.Sector == sector) & (df.TechnologyAdvancement == tech)
                   & (df.Flexibility == flex_e)].LoadMW)

    #print(np.sum(energy_b))
    #print(np.sum(energy_e))

    return energy_b, energy_e



def read_energy():
    fname = 'data/energy.csv'
    df = pd.read_csv(fname)
    print(df.columns)
    print(df.SCENARIO.unique())
    print(df.YEAR.unique())
    print(df.SECTOR.unique())
    #print(df.SUBSECTOR.unique())
    print(df.FINAL_ENERGY.unique())
    print(df[df.SECTOR == 'RESIDENTIAL'].SUBSECTOR.unique())
    print(df[df.SECTOR == 'COMMERCIAL'].SUBSECTOR.unique())
    print(df[df.SECTOR == 'TRANSPORTATION'].SUBSECTOR.unique())

    #print(df[(df.FINAL_ENERGY == 'SOLAR ') & (df.MMBTU > 0)].SUBSECTOR.unique())
    #print(df[(df.FINAL_ENERGY == 'SOLAR ') & (df.MMBTU > 0)].SCENARIO.unique())

    """
    subsector types:
    'RESIDENTIAL SPACE HEATING'
    'RESIDENTIAL WATER HEATING'
    'RESIDENTIAL COOKING'
    'RESIDENTIAL CLOTHES DRYING'
    'RESIDENTIAL SECONDARY HEATING'
    'COMMERCIAL SPACE HEATING'
    'COMMERCIAL WATER HEATING'
    'COMMERCIAL COOKING'
    """
    # ENTER
    #scenario = 'LOW ELECTRICITY GROWTH - MODERATE TECHNOLOGY ADVANCEMENT'
    #scenario = 'MEDIUM ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT'
    scenario = 'HIGH ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT'
    year_base = 2018
    energy = 'ELECTRICITY'

    #sector = 'RESIDENTIAL'
    sector = 'COMMERCIAL'
    sector_ev = 'TRANSPORTATION'

    sub_space = sector + ' SPACE HEATING'
    sub_water = sector + ' WATER HEATING'
    sub_ac = sector + ' AIR CONDITIONING'
    #sub_water = 'COMMERCIAL OTHER'
    sub_ev1 = 'LIGHT DUTY AUTOS'
    sub_ev2 = 'LIGHT DUTY TRUCKS'
    year = 2020
    state = 'CALIFORNIA'
    #state = 'VERMONT'
    #state = 'ARIZONA'
    #state = 'IOWA'

    base = df[(df.YEAR == year_base) & (df.STATE == state) & (df.SECTOR == sector)
              & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU
    base_space = df[(df.YEAR == year_base) & (df.STATE == state) & (df.SECTOR == sector) & (df.SUBSECTOR == sub_space)
                    & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU
    base_water = df[(df.YEAR == year_base) & (df.STATE == state) & (df.SECTOR == sector) & (df.SUBSECTOR == sub_water)
                    & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU
    base_ev1 = np.nan_to_num(df[(df.YEAR == year_base) & (df.STATE == state) & (df.SECTOR == sector_ev) & (df.SUBSECTOR == sub_ev1)
                    & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU)
    base_ev2 = np.nan_to_num(df[(df.YEAR == year_base) & (df.STATE == state) & (df.SECTOR == sector_ev) & (df.SUBSECTOR == sub_ev2)
                  & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU)

    us_base = df[(df.YEAR == year_base) & (df.SECTOR == sector)
              & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU
    us = df[(df.YEAR == year) & (df.SECTOR == sector)
              & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU

    all = df[(df.YEAR == year) & (df.STATE == state) & (df.SECTOR == sector)
              & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU
    space = df[(df.YEAR == year) & (df.STATE == state) & (df.SECTOR == sector) & (df.SUBSECTOR == sub_space)
                    & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU
    water = df[(df.YEAR == year) & (df.STATE == state) & (df.SECTOR == sector) & (df.SUBSECTOR == sub_water)
               & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU
    ac = np.nan_to_num(df[(df.YEAR == year) & (df.STATE == state) & (df.SECTOR == sector) & (df.SUBSECTOR == sub_ac)
               & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU)
    ev1 = np.nan_to_num(df[(df.YEAR == year) & (df.STATE == state) & (df.SECTOR == sector_ev) & (df.SUBSECTOR == sub_ev1)
                  & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU)
    ev2 = np.nan_to_num(df[(df.YEAR == year) & (df.STATE == state) & (df.SECTOR == sector_ev) & (df.SUBSECTOR == sub_ev2)
             & (df.SCENARIO == scenario) & (df.FINAL_ENERGY == energy)].MMBTU)


    #print(np.sum(all) / np.sum(base))
    #print(np.sum(space) / np.sum(all))
    #print(np.sum(water) / np.sum(all))
    #print(np.sum(base_space) / np.sum(base))
    #print(np.sum(base_water) / np.sum(base))
    #print((np.sum(space) / np.sum(all)) - np.sum(base_water) / np.sum(base))
    # the increase in space heating electricity over the baseline as a percentage of total electricity in the sim year
    print((np.sum(space) - np.sum(base_space)) / np.sum(all))  # this is the desired value
    #print((np.sum(space) - np.sum(base_space)) / np.sum(base))

    print((np.sum(water) - np.sum(base_water)) / np.sum(all))  # this is the desired value
    #print((np.sum(water) - np.sum(base_water)) / np.sum(base))
    other = np.sum(all) - np.sum(water) - np.sum(space)
    base_other = np.sum(base) - np.sum(base_water) - np.sum(base_space)
    print((other - base_other) / np.sum(all))  # this is the desired value
    #print((other - base_other) / np.sum(base))
    print('ratio of energy from ac', np.sum(ac) / np.sum(all))
    print(np.sum(water)/np.sum(all))
    print(np.sum(space)/np.sum(all))
    print('total MMBTU', np.sum(all))
    print('total MWh', np.sum(all) * 0.293071)

    # ENTER
    total_e = 24 * 365 * 0.5 / 1000 * 28
    com_percent = 1.0
    solar_percent = 0.115
    e_percent = ((np.sum(space) - np.sum(base_space)) / np.sum(all) + (np.sum(water) - np.sum(base_water)) / np.sum(all)
                 + com_percent * 0.2)
    # print('EV % relative to res+com energy use (2x res use)')
    print((np.sum(ev1 + ev2) - np.sum(base_ev1 + base_ev2)) / 2 / np.sum(all))
    print(e_percent)
    # print('energy by category:')
    storage_percent = solar_percent / 2.1
    print(total_e)
    # energy from electrification
    e_e = total_e * e_percent
    print(e_e)
    # energy from EVs
    ev_e = (total_e + e_e) * (np.sum(ev1 + ev2) - np.sum(base_ev1 + base_ev2)) / 2 / np.sum(all)
    print(ev_e)
    # energy from solar
    print((total_e + e_e + ev_e) * solar_percent)
    # energy in storage
    print((total_e + e_e + ev_e) * storage_percent)


def read_electrification(state, sector, year):
    fname = 'data/EFSLoadProfile_High_Moderate.csv'
    df = pd.read_csv(fname)
    #print(df.columns)
    #print(df.Electrification.unique())
    #print(df.TechnologyAdvancement.unique())
    #print(df.Year.unique())
    #print(df.State.unique())
    print(df.Sector.unique())
    #print(df[df.Sector == 'Residential'].Subsector.unique())
    #print(df[df.Sector == 'Commercial'].Subsector.unique())

    print(year)
    sector_ev = 'Transportation'
    sub_ev1 = 'light-duty vehicles'

    year_base = 2018

    com_all = np.sum(df[(df.Year == year) & (df.State == state) & (df.Sector == 'Commercial')].LoadMW)

    res_base = np.sum(df[(df.Year == year_base) & (df.State == state) & (df.Sector == sector)].LoadMW)
    res_space_base = np.sum(df[(df.Year == year_base) & (df.State == state) & (df.Sector == sector)
                       & (df.Subsector == 'space heating and cooling')].LoadMW)
    res_water_base = np.sum(df[(df.Year == year_base) & (df.State == state) & (df.Sector == sector)
                               & (df.Subsector == 'water heating')].LoadMW)
    res_other_base = res_base - res_water_base - res_space_base

    res_all = np.sum(df[(df.Year == year) & (df.State == state) & (df.Sector == sector)].LoadMW)
    res_space = np.sum(df[(df.Year == year) & (df.State == state) & (df.Sector == sector)
                  & (df.Subsector == 'space heating and cooling')].LoadMW)
    res_water = np.sum(df[(df.Year == year) & (df.State == state) & (df.Sector == sector)
                          & (df.Subsector == 'water heating')].LoadMW)
    res_other = res_all - res_water - res_space

    base_ev1 = np.sum(np.nan_to_num(
        df[(df.Year == year_base) & (df.State == state) & (df.Sector == sector_ev) & (df.Subsector == sub_ev1)].LoadMW))

    ev1 = np.sum(np.nan_to_num(
        df[(df.Year == year) & (df.State == state) & (df.Sector == sector_ev) & (df.Subsector == sub_ev1)].LoadMW))

    ev_ratio = (ev1 - base_ev1) / (com_all + res_all)

    print('ratio of new ev energy to baseline total energy', ev_ratio)
    print('ratio of new space energy to baseline total energy', (res_space - res_space_base) / res_base)
    print('ratio of new water energy to baseline total energy', (res_water - res_water_base) / res_base)
    print('ratio of new other energy to baseline total energy', (res_other - res_other_base) / res_base)
    print('ratio of new all energy to baseline total energy', (res_all - res_base) / res_base)

    return ev_ratio, (res_space - res_space_base) / res_base, (res_water - res_water_base) / res_base, (
                res_other - res_other_base) / res_base, res_all


def solar_storage_pen(pen2050):
    years = np.array([2020, 2024, 2030, 2035, 2040, 2045, 2050])
    solar_p_2050 = np.array([11.62040413, 16.04538541, 38.18254016, 58.68368254, 74.92921437, 87.89099572, 100])
    storage_p_solar = np.array([40, 45.2, 53, 59, 65, 67.5, 70])

    solar = solar_p_2050 * pen2050 / 100

    return years, solar, storage_p_solar



if __name__ == '__main__':
    """
        2050 solar penetration percent per city
        tracy	16.33
        vermont	13.11
        arizona	20.24
        central SF	11.5
        commercial SF	11.5
        oakland	11.5
        los banos	16.33
        marin	11.5
        rural san benito	16.33
        sacramento	16.33
        iowa	8.28
    """
    # name = 'tracy/'
    # name = 'rural_san_benito/'
    name = 'vermont/'
    solar_pen = 13.11
    # state = 'CA'
    state = 'VT'
    sector = 'Residential'
    # sector = 'Commercial'

    years, solar, storage = solar_storage_pen(solar_pen)

    year_base = 2018
    # years = np.array([2020, 2024, 2030, 2035, 2040, 2045, 2050])
    evs = []
    spaces = []
    waters = []
    others = []
    alls = []
    flex_bs = []
    flex_es = []
    for year in years:
        if year == 2035 or year == 2045:
            evs.append(np.nan)
            spaces.append(np.nan)
            waters.append(np.nan)
            others.append(np.nan)
            alls.append(np.nan)
            flex_bs.append(np.nan)
            flex_es.append(np.nan)
            continue

        ev, space, water, other, all = read_electrification(state, sector, year)
        flex_b, flex_e = read_flex(state, sector, year)
        evs.append(100 * ev)
        spaces.append(100 * space)
        waters.append(100 * water)
        others.append(100 * other)
        alls.append(all)
        flex_bs.append(100 * flex_b / all)
        flex_es.append(100 * flex_e / all)
        print(flex_bs)
        print(flex_es)

    pens = pd.DataFrame(
        {'year': years, 'solar': solar, 'storage': storage, 'ev': evs, 'space': spaces, 'water': waters, 'other': others,
         'flexibility base': flex_bs, 'flexibility enhanced': flex_es})

    pens.to_csv(name + sector + '_' + 'penetrations.csv')
    # NOTE:
    # Years 2025, 2035, and 2045 are not in the dataset. They must be interpolated from the data.

