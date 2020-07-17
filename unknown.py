import database as db


def prompt_unknown(num_unknown, unk_descriptors, database_name):
    """
    Prompts the user to name unknown faces
    :param num_unknown: the number of unknown faces
    :param unk_descriptors: the list of unknown descriptors
    :param database_name: the path of the database
    :return: nothing
    """
    for num in range(num_unknown):
        print(f"Enter the name of Unknown{num}:")
        name = str(input())
        db.add_profile(name, unk_descriptors[num], database_name)