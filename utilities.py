import pickle
import zipfile
import os
import hashlib
import IPython


def get_conditional_filename_hashing(filename, tolerance = 0):
    max_filename_length = os.pathconf('/', 'PC_NAME_MAX')
    #max_path_length = os.pathconf('/', 'PC_PATH_MAX')
    if len(filename) > max_filename_length - tolerance:# or len(os.path.abspath(filename)) > max_path_length:
        hashed_filename = hashlib.sha256(filename.encode()).hexdigest()
        return hashed_filename
    return filename


def save_file(filename, data):
    max_filename_length = os.pathconf('/', 'PC_NAME_MAX')
    max_path_length = os.pathconf('/', 'PC_PATH_MAX')

    if len(filename) > max_filename_length or len(os.path.abspath(filename)) > max_path_length:
        hashed_filename = hashlib.sha256(filename.encode()).hexdigest()
        with open(hashed_filename, 'wb') as file:
            file.write(data)
        return hashed_filename
    else:
        with open(filename, 'wb') as file:
            file.write(data)
        return filename



def produce_parallelism_schedule(num_exps, max_parallelism):
  counter = 0
  remainder = num_exps
  result = []
  while counter < num_exps:
    curr_batch = min(max_parallelism, remainder)
    counter += curr_batch
    remainder -= curr_batch
    result.append(curr_batch)

  #IPython.embed()
  if sum(result) != num_exps:
    raise ValueError("Sum of parallelism schedule != num_exps.")
  if min(result) == 0:
    raise ValueError("Minimum parallelims batch equals 0.")
  
  return result


def pickle_and_zip(obj, results_filename_stub, base_data_dir, is_zip_file = False, hash_filename  = False):
  if hash_filename:
    processed_results_filename_stub = get_conditional_filename_hashing(results_filename_stub)
  processed_results_filename_stub = results_filename_stub

  pickle_results_filename = "{}.p".format(processed_results_filename_stub)
  ### start by saving the file using pickle

  pickle.dump( obj, 
    open("{}/{}".format(base_data_dir, pickle_results_filename), "wb"))

  if is_zip_file:

    zip_results_filename = "{}.zip".format(processed_results_filename_stub)
    zip_file = zipfile.ZipFile("{}/{}".format(base_data_dir, zip_results_filename), 'w')

    zip_file.write("{}/{}".format(base_data_dir, pickle_results_filename), compress_type = zipfile.ZIP_DEFLATED, 
      arcname = os.path.basename("{}/{}".format(base_data_dir, pickle_results_filename)) )
    
    zip_file.close()

    os.remove("{}/{}".format(base_data_dir, pickle_results_filename))


def unzip_and_load_pickle(base_data_dir, results_filename_stub, is_zip_file = False, hash_filename = False):
  if hash_filename:
    processed_results_filename_stub = get_conditional_filename_hashing(results_filename_stub)
  processed_results_filename_stub = results_filename_stub

  pickle_results_filename = "{}.p".format(processed_results_filename_stub)

  ## If it is a ZIP file extract the pickle file.
  if is_zip_file:
    zip_results_filename = "{}.zip".format(processed_results_filename_stub)
    
    
    zip_file = zipfile.ZipFile("{}/{}".format(base_data_dir, zip_results_filename), "r")
    zip_file.extractall(base_data_dir)

  results_dictionary = pickle.load( open("{}/{}".format(base_data_dir, pickle_results_filename), "rb") )

  ## If it is a ZIP file, delete the pickle file.
  if is_zip_file:
    os.remove("{}/{}".format(base_data_dir, pickle_results_filename))

  return results_dictionary



def write_dictionary_file(dictionary, filename):
  with open(filename, "w") as f:
    for a in dictionary.keys():
      b = dictionary[a]
      f.write("{} {}".format(a,b))
      f.write('\n')
    f.close()




def read_dictionary_file(filename):
  dictionary = dict([])
  with open(filename, "r") as f:
    for line in f.readlines():

      algo_name = line.split(" ")[0]
      
      mean_val = int(line.split(" ")[1][1:-1])
      std_val = int(line.split(" ")[2][:-2])
      dictionary[algo_name] = (mean_val, std_val)
    f.close()
  return dictionary





### the assumption is 
def write_tuple_file(tuple_data, filename):
  with open(filename, "w") as f:
    text = str(tuple_data[0])

    for i in range(1,len(tuple_data)):
      text += " {}".format(tuple_data[i])
    f.write(text)
    f.close()



### the assumption is the input is of length 3
def read_tuple_file(filename):
  list_result = []
  with open(filename, "r") as f:
    line = f.readlines()[0]

    elements = line.split(" ")
    algo_name = elements[0]
    mean_val = int(elements[1])
    std_val = int(elements[2])
  f.close()
  return (algo_name, mean_val, std_val)


