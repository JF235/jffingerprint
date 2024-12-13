import os
import shutil

def flatten_directory(directory_path):
    # Verifica se o diretório existe
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    # Cria o novo diretório com o sufixo "_flat"
    flat_directory_path = directory_path + "_flat"
    if not os.path.exists(flat_directory_path):
        os.makedirs(flat_directory_path)

    # Percorre todos os arquivos e subdiretórios no diretório original
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Move cada arquivo para o novo diretório "_flat"
            file_path = os.path.join(root, file)
            shutil.move(file_path, flat_directory_path)

    print(f"All files have been moved to {flat_directory_path}")

if __name__ == "__main__":
    directory_path = input("Enter the path of the directory to flatten: ")
    flatten_directory(directory_path)