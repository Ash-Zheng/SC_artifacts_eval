import hugectr
from hugectr.tools import DataGenerator, DataGeneratorParams
from mpi4py import MPI


# Parquet data generation
data_generator_params = DataGeneratorParams(
  format = hugectr.DataReaderType_t.Parquet,
  label_dim = 1,
  dense_dim = 1,
  num_slot = 1,
  num_files = 4,
  eval_num_files = 4,
  i64_input_key = True,
  source = "/workspace/HugeCTR/HugeCTR_data/file_list.txt",
  eval_source = "/workspace/HugeCTR/HugeCTR_data/file_list_test.txt",
  slot_size_array = [40000000],
  # for parquet, check_type doesn't make any difference
  check_type = hugectr.Check_t.Non,
  dist_type = hugectr.Distribution_t.PowerLaw,
  power_law_type = hugectr.PowerLaw_t.Short)
data_generator = DataGenerator(data_generator_params)
data_generator.generate()