import pandas as pd
import xport

if __name__ == "__main__":
    days = {1: 'sunday', 2: 'monday', 3: 'tuesday', 4: 'wednesday', 5: 'thursday', 6: 'friday', 7: 'sunday'}
    timestep = 5  # sampling interval from run_sim.m
    meals = {1: 'breakfast', 2: 'lunch', 3: 'dinner', 4: 'dinner', 5: 'breakfast', 6: 'snack', 7: 'drink',
             8: 'infant', 9: 'snack', 10: 'breakfast', 11: 'lunch', 12: 'dinner', 13: 'lunch', 14: 'dinner',
             15: 'snack', 16: 'snack', 17: 'lunch', 18: 'snack', 19: 'drink', 91: 'other', 99: 'idk'}


    full_df = pd.read_sas('./data/DR1IFF_J.XPT')
    df = full_df[['SEQN', "DR1DAY", "DR1_020", 'DR1_030Z', 'DR1ICARB']]  # Day, Time (HH:MM), Meal type,  carb amount

    unique = pd.unique(df['SEQN'])
    ids = {unique[i]: i for i in range(len(unique))}

    df = df.rename(columns={'SEQN': 'id', "DR1DAY": 'day', "DR1_020": 'time', 'DR1_030Z': 'meal_type', 'DR1ICARB': 'carb'})


    df['id'] = df['id'].map(lambda x: ids[x])
    df['day'] = df['day'].map(lambda x: days[x])  # convert to day of week
    df['time'] = df['time'].map(lambda x: x // 60 // timestep)  # convert to time (mins / sampling interval)
    df['meal_type'] = df['meal_type'].map(lambda x: meals[x])

    print('done')

