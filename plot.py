#!/usr/bin/python3

# File:     A program which parses the output data from the test_ops, and creates plots.
#           The code is derived from my matrix-multiplication project.
#
# Author:   Ardalan Ahanchi
# Date:     Winter 2020

import re                               #For regex pattern recognition.
import sys                              #To read from stdin, and argv
import matplotlib.pyplot as plot        #For plotting.

#Define all the possible integer keys.
int_keys = ["rows", "cols"]

#Define all the possible float keys.
float_keys = ["time"]

#Define all the possible string keys.
str_keys = ["mode", "op"]

#Create a regex pattern for matching the format for the data from the output.
regex_pattern = '\[(\S+)\]=(\S+)'

#Create an array for holding the parsed data entries.
entries = []

#Iterate through every line in stdin to read data.
for line in sys.stdin:
    #Define a dictionary for the values.
    entry = {}

    #Go through all the matches, and add them to the data dictionary.
    for match in re.finditer(regex_pattern, line):
        #Check if the key's value should be converted to an integer.
        if match.group(1) in int_keys:
            entry[match.group(1)] = int(match.group(2))

        #Check if the key's value should be converted to a floating point.
        if match.group(1) in float_keys:
            entry[match.group(1)] = float(match.group(2))

        #Check if the key's value should be stored as a string.
        if match.group(1) in str_keys:
            entry[match.group(1)] = str(match.group(2))
        
    #Add the current entry (dict) into the list of entries.
    entries.append(entry)


#A function which compares the cpu, gpu, and hybrid versions of the op mode passed to it.
def examine_op(op, op_string):
    #Define dictopmaroes for holding the relevant data key: mode, res: list of sizes.
    times_sq = {"cpu" : [], "gpu" : [], "hybrid" : []}    #For square data (rows == cols).
    sizes_sq = {"cpu" : [], "gpu" : [], "hybrid" : []}   
    
    times_vc = {"cpu" : [], "gpu" : [], "hybrid" : []}    #For vectors (rows == 1).
    sizes_vc = {"cpu" : [], "gpu" : [], "hybrid" : []}   

    #Go through all data entries and populate the lists.    
    for ent in entries:
        #Filter the rest based on the passed opmode.
        if ent["op"] == op:
            #Check if we're comparing square data.
            if ent["rows"] == ent["cols"]:
                #Append data to the correct mode.
                times_sq[ent["mode"]].append(ent["time"])
                sizes_sq[ent["mode"]].append(ent["cols"])
            
            #If we're comparing vector data.
            elif ent["rows"] == 1:
                #Append data to the correct mode.
                times_vc[ent["mode"]].append(ent["time"])
                sizes_vc[ent["mode"]].append(ent["cols"])

    # Figure the square matrices results #######################################
    #For every mode, plot the results.
    for curr_mode, curr_time in times_sq.items():
        plot.plot(sizes_sq[curr_mode], times_sq[curr_mode], label=curr_mode)

    #Add a legend, titles, and draw the plot.
    plot.legend()
    plot.title("Matrix " + op_string + " operations on different modes.")
    plot.xlabel("N (Size of the matrix is NxN)")
    plot.ylabel("Time (Seconds)")
    plot.show()

    # Figure the vector results ###############################################
    #For every mode, plot the results.
    for curr_mode, curr_time in times_vc.items():
        plot.plot(sizes_vc[curr_mode], times_vc[curr_mode], label=curr_mode)

    #Add a legend, titles, and draw the plot.
    plot.legend()
    plot.title("Vector " + op_string + " operations on different modes.")
    plot.xlabel("N (Size of the matrix is 1xN)")
    plot.ylabel("Time (Seconds)")
    plot.show()
        

def main():
    #Start the examination for all operations.
    examine_op("add", "addition")
    examine_op("sub", "subtraction")
    examine_op("e_mult", "element multiplication")
    examine_op("mult", "multiplication")
    examine_op("scale", "scaling")
    examine_op("sigmoid", "Sigmoid")
    examine_op("deriv_sigmoid", "Sigmoid Prime")
    examine_op("relu", "ReLu")
    examine_op("deriv_relu", "ReLu Prime")

if __name__ == "__main__":
    main()
