def print_cpp_vector(vector):
    print("std::vector<float> expected = {", end="")
    for i in range(len(vector)):
        if i != 0:
            print(", ", end="")
        print(str(round(vector[i].item(), 5)) + "f", end="")
    print("};")
