# Author: Ali Barış Bilen 
# Date: 13.02.2022

# This little program will help you split your pdf's fast & safely.
# You no more have to upload your pdf's to untrustable websites that potentially process your input data outside the country you're living in.

# Usage: Please read usage.txt located in program_files folder.

from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
import os
import getpass
from colorama import *
from termcolor import colored
import sys
from pyfiglet import Figlet 

init(autoreset=True)
user_name = getpass.getuser()

if __name__ == "__main__":

    figlet_object = Figlet(font = 'standard')
    print(colored(figlet_object.renderText("Welcome to PDF Splitter"),'green'))
    print(1*"\n")
    print(colored("Author: Ali Barış Bilen",'blue','on_white', attrs=['bold']))
    print(colored("Contact: {mynameandsurnameinlowercase}1@gmail.com",'blue','on_white', attrs=['bold']))
    print(1*"\n")
    
    main_path = "C:/Users/" + user_name + "/Desktop/PdfSplitter/"
    files_to_be_split = [f for f in os.listdir(main_path + "/files_to_be_split/") if f.endswith('.pdf')]

    pdf_file_count = len(files_to_be_split)

    if pdf_file_count == 0:
        print(colored("There isn't any pdf files in the folder!\nPlease check again.", 'yellow'))
        sys.exit()
    else:
        print(colored("There are " + str(pdf_file_count) + " pdf files in the folder.",'yellow'))
        print(colored("Those are: ",'yellow'))
        print(*files_to_be_split, sep = " | ")
        file_index = int(input("Which file do you want to split, please enter an index: ")) - 1
        while (file_index + 1) not in range(1, pdf_file_count + 1):
            print(colored("Wrong input.\nThere are only " + str(pdf_file_count) + " files in folder.",'red'))
            print(colored("Please enter the index again.",'yellow'))
            file_index = int(input("Enter: ")) - 1

    temp_file = files_to_be_split[file_index]

    input_pdf = PdfFileReader(main_path + "/files_to_be_split/" + temp_file)
    
    # determining pages

    from_page = int(input("Please enter page number to split, from:   "))
    to_page = int(input("Please enter page number to split, to:   "))

    os.mkdir("temp_file_folder_for_" + temp_file[:len(temp_file)-4])
    os.chdir("temp_file_folder_for_" + temp_file[:len(temp_file)-4])

    for i in range(from_page,to_page + 1):
        output = PdfFileWriter()
        output.addPage(input_pdf.getPage(i-1))
        output_name = str(temp_file + '_page_' + str(i) + ".pdf")
        with open(output_name,"wb") as output_stream:
            output.write(output_stream)

    # merging

    file_list = list(os.listdir())
    main_output = PdfFileMerger()

    for j in file_list:
        temp_pdf = PdfFileReader(j)
        main_output.append(temp_pdf)
        os.remove(j)

    os.chdir(main_path)

    if os.path.exists("program_files/temp_file_folder_for_" + temp_file[:len(temp_file)-4] + "/"):
        os.rmdir("program_files/temp_file_folder_for_" + temp_file[:len(temp_file)-4] + "/")
        # why not shutil()? - because shutil() operations are irreversible (it's too dangerous and I don't prefer using it here).

    main_output_temp = str(input("Please enter a name for merged pdf output (just press enter to keep the same file name): "))
    
    if len(main_output_temp) == 0:
        main_output_name = str(temp_file[:len(temp_file)-4] + "_v2" + ".pdf")
    else:
        main_output_name = str(main_output_temp + ".pdf")

    if not os.path.exists('Output'):
        os.makedirs('Output')

    os.chdir("Output")

    with open(main_output_name, "wb") as output_stream:
        main_output.write(output_stream)

    print(colored("New PDF is successfully created. Please find it in the 'Output' folder.",'green'))
