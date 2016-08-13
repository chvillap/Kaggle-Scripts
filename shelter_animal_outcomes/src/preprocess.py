import pandas as pd
import numpy as np
from datetime import datetime as dt
import re
from sklearn.preprocessing import LabelEncoder

#______________________________________________________________________________

def preprocess_name(dataset):
    """Preprocesses the "Name" attribute, changing it to a "HasName"
    attribute which is False for every unnamed animal and True for all others.
    """
    dataset['HasName'] = dataset['Name'].notnull()

    return dataset


#______________________________________________________________________________

def preprocess_datetime(dataset):
    """Preprocesses the "DateTime" attribute, splitting it into five others:
    "Year", "Month", "Day", "Weekday" and "Hour".
    """
    datetime = dataset['DateTime'].apply(
        lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))

    dataset['Year'] = datetime.apply(lambda x: x.year)
    dataset['Month'] = datetime.apply(lambda x: x.month)
    dataset['Day'] = datetime.apply(lambda x: x.day)
    dataset['Weekday'] = datetime.apply(lambda x: x.weekday())
    dataset['Hour'] = datetime.apply(lambda x: x.hour)

    return dataset


#______________________________________________________________________________

def preprocess_sex(dataset):
    """Preprocesses the "SexUponOutcome" attribute, splitting it into two
    others: "Sex" (self-explanatory) and "Sterile", which states whether an
    animal is neutered/spayed or not.
    """
    sterile_sex = dataset['SexuponOutcome'].str.split(' ', expand=True)

    sterile_sex[0] = sterile_sex[0].str.replace(r'Neutered|Spayed', 'True')
    sterile_sex[0] = sterile_sex[0].str.replace(r'Intact', 'False')

    sterile_sex.ix[sterile_sex[0] == 'Unknown', 1] = 'Unknown'

    dataset['Sterile'] = sterile_sex[0].fillna('Unknown')
    dataset['Sex'] = sterile_sex[1].fillna('Unknown')

    return dataset


#______________________________________________________________________________

def preprocess_age(dataset):
    """Preprocesses the "AgeUponOutcome" attribute, changing it to an
    "AgeInDays" attribute which counts the ages in days.
    """
    def transform_age(x):
        if pd.isnull(x):
            return x
        age_unit = x.split(' ')
        if 'year' in age_unit[1]:
            return int(age_unit[0]) * 365
        if 'month' in age_unit[1]:
            return int(age_unit[0]) * 30
        if 'week' in age_unit[1]:
            return int(age_unit[0]) * 7
        if 'day' in age_unit[1]:
            return int(age_unit[0])

    dataset['AgeInDays'] = dataset['AgeuponOutcome'].apply(transform_age)

    return dataset


#______________________________________________________________________________

def preprocess_breed(dataset):
    """Preprocesses the "Breed" attribute, splitting it in two others:
    "PrimaryBreed" (self-explanatory) and "SecondaryBreed", which has no value
    for pure breed animals, values null for multiple breed ('Mix') animals, or
    has some other breed name for animals with exactly two breeds.
    Moreover, inconsistencies in breed names are fixed and three additional
    attributes are created: "BreedGroup", "BodySize" and "HairLength", which
    describe particular features of the breeds. 
    """
    def get_breedgroup(x):
        breedname = breed_stats['BreedName']
        breedgroup = breed_stats['BreedGroup']
        result = breedgroup[breedname == x].tolist()
        return None if not result else result[0]

    def get_bodysize(x):
        breedname = breed_stats['BreedName']
        bodysize = breed_stats['BodySize']
        result = bodysize[breedname == x].tolist()
        return None if not result else result[0]

    def get_hairlength(x):
        breedname = breed_stats['BreedName']
        hairlength = breed_stats['HairLength']
        result = hairlength[breedname == x].tolist()
        return None if not result else result[0]

    breed_stats = pd.read_csv('datasets/aux/breed_stats.csv')

    breed = dataset['Breed'] \
        .str.replace('Anatol Shepherd', 'Anatolian Shepherd Dog') \
        .str.replace('Bedlington Terr', 'Bedlington Terrier') \
        .str.replace('Black/Tan Hound', 'Black and Tan Coonhound') \
        .str.replace('Bluetick Hound', 'Bluetick Coonhound') \
        .str.replace('Boykin Span', 'Boykin Spaniel') \
        .str.replace('Bruss ', 'Brussels ') \
        .str.replace('Catahoula', 'Catahoula Cur') \
        .str.replace('Cavalier Span', 'Cavalier King Charles Spaniel') \
        .str.replace('Chesa Bay Retr', 'Chesapeake Bay Retriever') \
        .str.replace('Collie Smooth', 'Smooth Collie') \
        .str.replace('Collie Rough', 'Rough Collie') \
        .str.replace('Dachshund Stan', 'Dachshund') \
        .str.replace('Doberman Pinsch', 'Doberman Pinscher') \
        .str.replace('Eng Toy Spaniel', 'English Toy Spaniel') \
        .str.replace('English Coonhound', 'American English Coonhound') \
        .str.replace('Entlebucher', 'Entlebucher Mountain Dog') \
        .str.replace('Eskimo', 'Eskimo Dog') \
        .str.replace('German Shorthair', 'German Shorthaired') \
        .str.replace('Mexican Hairless', 'Xoloitzcuintli') \
        .str.replace('Of Imaal', 'of Imaal Terrier') \
        .str.replace('Patterdale Terr', 'Patterdale Terrier') \
        .str.replace('Port Water', 'Portuguese Water') \
        .str.replace('Pbgv', 'Petit Basset Griffon Vendeen') \
        .str.replace('Picardy Sheepdog', 'Berger Picard') \
        .str.replace('Presa Canario', 'Perro de Presa Canario') \
        .str.replace('Podengo', 'Portuguese Podengo') \
        .str.replace('Queensland Heeler', 'Australian Cattle Dog') \
        .str.replace('Redbone Hound', 'Redbone Coonhound') \
        .str.replace('Rhod Ridgeback', 'Rhodesian Ridgeback') \
        .str.replace('Schnauzer Giant', 'Giant Schnauzer') \
        .str.replace('Sealyham Terr', 'Sealyham Terrier') \
        .str.replace('Sharpei', 'Shar Pei') \
        .str.replace('West Highland', 'West Highland White Terrier') \
        .str.replace('Wire Hair Fox', 'Wire Fox') \
        .str.replace('Oriental Sh', 'Oriental Shorthair') \
        .str.replace('Pixiebob Shorthair', 'Pixiebob') \
        .str.replace('Swiss Hound', 'Schweizer Laufhund') \
        .str.replace(r'(Flat\sCoat)[^e/$]', r'\g<1>ed ') \
        .str.replace(r'(^|/)(Angora)', r'\g<1>Turkish \g<2>') \
        .str.replace(r'(^|/)(Bulldog|Pointer)', r'\g<1>English \g<2>') \
        .str.replace(r'(^|/)(Staffordshire)(\sMix|/|$)', r'\g<1>\g<2> Bull Terrier\g<3>') \
        .str.replace(r'(Yorkshire|Dandie\sDinmont)(/|$)', r'\g<1> Terrier\g<2>') \
        .str.replace(r'(^|/)(Rex)', r'\g<1>Cornish \g<2>')

    prim_sec = breed.str.split('/', expand=True)
    prim_breed = prim_sec[0].str.replace(' Mix', '')
    sec_breed = prim_sec[1].str.replace(' Mix', '')

    prim_ismix = prim_sec[0].apply(lambda x: ' Mix' in x)
    sec_isnull = sec_breed.isnull()
    sec_breed[prim_ismix & sec_isnull] = 'Mix'

    breedcount = prim_breed.value_counts()
    is_rare = breedcount[breedcount <= 0.01 * breedcount.max()].index

    dataset['PrimBreed'] = prim_breed.replace(is_rare, 'Rare')
    dataset['PrimBreedGroup'] = prim_breed.apply(get_breedgroup)
    dataset['PrimBodySize'] = prim_breed.apply(get_bodysize)
    dataset['PrimHairLength'] = prim_breed.apply(get_hairlength)

    # dataset['SecBreed'] = sec_breed.fillna('None')
    # dataset['SecBreedGroup'] = sec_breed.apply(get_breedgroup).fillna('None')
    # dataset['SecBodySize'] = sec_breed.apply(get_bodysize).fillna('None')
    # dataset['SecHairLength'] = sec_breed.apply(get_hairlength).fillna('None')

    return dataset


#______________________________________________________________________________

def preprocess_color(dataset):
    """Preprocesses the "Color" attribute, splitting it in four others:
    "PrimColor", "PrimPattern", "SecColor" and "SecPattern". 'Tricolor' is
    considered a single color. 'None' values are used wherever they are
    appropriate.
    """
    def check_color_name(x):
        return x and x.lower() in valid_colors

    def empty_str_if_null(x):
        return '' if x is None else x

    color_stats = pd.read_csv('aux/color_stats.csv', error_bad_lines=True)
    valid_colors = set(color_stats['Name'].tolist())

    prim_sec = dataset['Color'].str.split('/', expand=True)
    prim_cp = prim_sec[0].str.split(' ', expand=True)
    sec_cp = prim_sec[1].str.split(' ', expand=True)

    iscolor = prim_cp[0].apply(check_color_name).fillna(True)
    prim_cp.ix[~iscolor,1] = prim_cp.ix[~iscolor,0].apply(empty_str_if_null) + \
                             prim_cp.ix[~iscolor,1].apply(empty_str_if_null)
    prim_cp.ix[~iscolor,0] = None

    iscolor = sec_cp[0].apply(check_color_name).fillna(True)
    sec_cp.ix[~iscolor,1] = sec_cp.ix[~iscolor,0].apply(empty_str_if_null) + \
                            sec_cp.ix[~iscolor,1].apply(empty_str_if_null)
    sec_cp.ix[~iscolor,0] = None

    dataset['PrimColor'] = prim_cp[0].fillna('None')
    dataset['PrimPattern'] = prim_cp[1].fillna('None')
    # dataset['SecColor'] = sec_cp[0].fillna('None')
    # dataset['SecPattern'] = sec_cp[1].fillna('None')

    return dataset


#______________________________________________________________________________

def drop_unused(dataset):
    """Drops attributes that are not useful for the sake of predicting outcomes
    for new animals.
    """
    columns = [
        'OutcomeSubtype', # comment this when running on the test dataset.
        'Name', 'DateTime', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 
        'Color']

    dataset.drop(columns, axis=1, inplace=True)

    return dataset


#______________________________________________________________________________

def encode_categorical(dataset):
    """Encode categorical attributes so that each value is transformed into an
    integer number. PS: maybe I should try one-hot encoding too.
    """
    categorical_col = [
        'OutcomeType', # comment this when running on the test dataset.
        'AnimalType', 'HasName', 'Sterile', 'Sex', 'PrimBreed',
        'PrimBreedGroup', 'PrimBodySize', 'PrimHairLength', 'PrimColor',
        'PrimPattern']

    le = LabelEncoder()
    for col in categorical_col:
        dataset[col] = le.fit_transform(dataset[col])

    return dataset


#______________________________________________________________________________

def fillna_age(dataset):
    """Fill missing values in 'AgeInDays' attribute using a regression method. 
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor

    categorical_col = [
        'OutcomeType', # comment this when running on the test dataset.
        'AnimalType', 'HasName', 'Sterile', 'Sex', 'PrimBreed',
        'PrimBreedGroup', 'PrimBodySize', 'PrimHairLength', 'PrimColor',
        'PrimPattern']
    X = dataset.drop(categorical_col, axis=1)

    is_age_null = X['AgeInDays'].isnull()
    X_train = X.drop(['AgeInDays'], axis=1)[~is_age_null]
    y_train = X.ix[~is_age_null, 'AgeInDays']

    reg = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100)
    reg.fit(X_train, y_train)

    X_pred = X.drop(['AgeInDays'], axis=1)[is_age_null]
    y_pred = reg.predict(X_pred)

    dataset.ix[is_age_null, 'AgeInDays'] = y_pred

    return dataset


#______________________________________________________________________________

if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3:
        print('Usage: {0} <dataset_in.csv> <dataset_out.csv>'.format(argv[0]))
    else:
        dataset_in = pd.read_csv(argv[1], index_col=0, error_bad_lines=True)
        
        dataset_out = preprocess_name(dataset_in)
        dataset_out = preprocess_datetime(dataset_out)
        dataset_out = preprocess_sex(dataset_out)
        dataset_out = preprocess_age(dataset_out)
        dataset_out = preprocess_breed(dataset_out)
        dataset_out = preprocess_color(dataset_out)
        dataset_out = drop_unused(dataset_out)
        dataset_out = encode_categorical(dataset_out)
        dataset_out = fillna_age(dataset_out)

        dataset_out.to_csv(argv[2], error_bad_lines=True)
