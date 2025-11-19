#!/usr/bin/env python
# coding: utf-8

# # Transpose the file

# In[1]:


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split('\t') for line in lines]

def write_file(file_path, data):
    with open(file_path, 'w') as file:
        for row in data:
            file.write('\t'.join(row) + '\n')

def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))

def main():
    input_file = "File_Name_To_Transpose.txt"
    output_file = "Output_Traspose.txt"

    # Read the data from the input file
    data = read_file(input_file)

    # Transpose the matrix
    transposed_data = transpose_matrix(data)

    # Write the transposed data to the output file
    write_file(output_file, transposed_data)

if __name__ == "__main__":
    main()


# # Data Dissociation

# In[2]:


def split_file_into_multiple_files(input_file, num_columns_per_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    total_columns = len(lines[0].split())
    num_files = total_columns // num_columns_per_file

    for file_num in range(num_files + 1):
        start_col = file_num * num_columns_per_file
        end_col = min((file_num + 1) * num_columns_per_file, total_columns)
        output_file = f'output_file_{file_num + 1}.txt'

        with open(output_file, 'w') as out_f:
            for line in lines:
                columns = line.split()
                selected_columns = columns[start_col:end_col]
                out_f.write('\t'.join(selected_columns) + '\n')


if __name__ == "__main__":
    input_file = "Output_Traspose.txt"
    num_columns_per_file = 10000
    split_file_into_multiple_files(input_file, num_columns_per_file)


# In[3]:


#Good luck


#Zarlish Attique.. 

