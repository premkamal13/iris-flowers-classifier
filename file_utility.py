#Utility class

def write_to_file(filename, data):
	f = open(filename, "w+")
	f.write(data)
