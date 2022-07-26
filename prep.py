# glob utilities
import glob

# string templates
import string

# set directory to prepare
DIR = "./data/AphasiaBankEnglishProtocol/*.txt" # (output of flo "+t*" +ca *.cha)
WINDOW = 10 # we will train with a window of 10 utterances

# search for all chat files
chat_files = glob.glob(DIR)

# read all chat files
def read_file(f):
    """Utility to read a single flo file

    Arguments:
        f (str): the file to read

    Returns:
        list[str] a string of results
    """
    
    # open and read file
    with open(f, 'r') as df:
        # read!
        lines = df.readlines()

    # coallate results
    results = []

    # process lines for tab-deliminated run-on lines
    for line in lines:
        # if we have a tab
        if line[0] == '\t':
            # take away the tab, append, and put back in results
            results.append(results.pop()+" "+line.strip())
        # otherwise, just append
        else:
            results.append(line.strip())

    # return results
    return results

# prep all the files
cleaned_files = sum([read_file(i) for i in chat_files], [])

with open("./data/AphasiaBankEnglishProtocol.txt", 'w') as df:
    df.writelines([i+'\n' for i in cleaned_files])


