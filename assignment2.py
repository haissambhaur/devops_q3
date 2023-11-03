import math
import random
from turtle import color, left, right
import matplotlib.pyplot as plt
import numpy as np

amount=10

lst = [0] * (10)
print("Do you want custom array?(Y/N):")
string=input()
if(string=='Y' or string=='y'):
    print("Enter 10 numbers for sorting less than 100")
    for i in range(amount):
        lst[i]=input()
elif(string=='N' or string=='n'):
    lst=np.random.randint(0,100,amount)
x = np.arange(0,amount,1) #x scale 
                #start, number of elements, step size
plt.bar(x,lst)
plt.pause(2)
plt.close()
size=10

def getFirstDigit(num):
    return int(num / 10 ** int(math.log10(num)))

def bubbleSort():
    

    for i in range(size):
        for j in range(0, size-i-1):
            plt.bar(x,lst,color=['green'])
            plt.pause(0.001)
            plt.clf()
            if(lst[j] > lst[j+1]):
                lst[j], lst[j+1] = lst[j+1], lst[j]
    plt.clf()
    plt.bar(x,lst,color=['green'])
    plt.show()


def insertionSort():
    lst=np.random.randint(0,100,amount)
    x = np.arange(0,amount,1) #x scale 
    for i in range(size):
        j=i
        while j>0:
            plt.bar(x,lst,color=['green'])
            plt.pause(0.001)
            plt.clf()
            if lst[j-1] > lst[j]:
                #swap
                lst[j-1],lst[j] = lst[j],lst[j-1]
            else:
                break
            j = j-1
    plt.clf()
    plt.bar(x,lst,color=['green'])
    plt.show()


def selectionSort():
    lst=np.random.randint(0,100,amount)
    x = np.arange(0,amount,1) #x scale 
    for i in range(size):
        min_idx = i
        for j in range(i + 1, size):
            plt.bar(x,lst,color=['green'])
            plt.pause(0.001)
            plt.clf()
            if lst[j] < lst[min_idx]:
                #swap
                min_idx=j
        (lst[i], lst[min_idx]) = (lst[min_idx], lst[i])
    plt.clf()
    plt.bar(x,lst,color=['green'])
    plt.show()

def merge_sort(list1, left_index, right_index):  
    if left_index >= right_index:  
        return  

    middle = (left_index + right_index)//2  
    plt.bar(list(range(amount)),list1,color=['red'])
    plt.pause(0.01)
    plt.clf()
    
    merge_sort(list1, left_index, middle)  
    merge_sort(list1, middle + 1, right_index)  

    plt.bar(list(range(amount)),list1,color=['red'])
    plt.pause(0.01)
    plt.clf()
    
    merge(list1, left_index, right_index, middle)  
    plt.bar(list(range(amount)),list1,color=['red'])
    plt.pause(0.01)
    plt.clf()
    
  
  
    # Defining a function for merge the list  
def merge(list1, left_index, right_index, middle):  
  
  
   # Creating subparts of a lists  
    left_sublist = list1[left_index:middle + 1]  
    right_sublist = list1[middle+1:right_index+1]  
  
    # Initial values for variables that we use to keep  
    # track of where we are in each list1  
    left_sublist_index = 0  
    right_sublist_index = 0  
    sorted_index = left_index  
  
    # traverse both copies until we get run out one element  
    while left_sublist_index < len(left_sublist) and right_sublist_index < len(right_sublist):  
  
        # If our left_sublist has the smaller element, put it in the sorted  
        # part and then move forward in left_sublist (by increasing the pointer)  
        if left_sublist[left_sublist_index] <= right_sublist[right_sublist_index]:  
            list1[sorted_index] = left_sublist[left_sublist_index]  
            left_sublist_index = left_sublist_index + 1  
        # Otherwise add it into the right sublist  
        else:  
            list1[sorted_index] = right_sublist[right_sublist_index]  
            right_sublist_index = right_sublist_index + 1  
  
  
        # move forward in the sorted part  
        sorted_index = sorted_index + 1  
  
       
    # we will go through the remaining elements and add them  
    while left_sublist_index < len(left_sublist):  
        list1[sorted_index] = left_sublist[left_sublist_index]  
        left_sublist_index = left_sublist_index + 1  
        sorted_index = sorted_index + 1  
  
    while right_sublist_index < len(right_sublist):  
        list1[sorted_index] = right_sublist[right_sublist_index]  
        right_sublist_index = right_sublist_index + 1  
        sorted_index = sorted_index + 1 

def count_sort(lst):
    
    plt.bar(x,lst)
    plt.pause(0.01)
    plt.clf()
    max_element = int(max(lst))
    min_element = int(min(lst))
    range_of_elements = max_element - min_element + 1
    count_arr = [0 for _ in range(range_of_elements)]
    output_arr = [0 for _ in range(len(lst))]
    plt.bar(x,lst)
    plt.pause(0.01)
    plt.clf() 
    for i in range(0, len(lst)):
        count_arr[lst[i]-min_element] += 1

    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i-1]
  
    for i in range(len(lst)-1, -1, -1):
        output_arr[count_arr[lst[i] - min_element] - 1] = lst[i]
        count_arr[lst[i] - min_element] -= 1
  
    for i in range(0, len(lst)):
        lst[i] = output_arr[i]
        plt.bar(x,lst)
        plt.pause(0.01)
        plt.clf()
    plt.clf()
    plt.bar(x,lst)
    plt.show()


def radixSort1(arr, exp1):
 
    n = len(arr)
 
    # The output array elements that will have sorted arr
    output = [0] * (n)
 
    # initialize count array as 0
    count = [0] * (10)
 
    # Store count of occurrences in count[]
    for i in range(0, n):
        index = arr[i] // exp1
        count[index % 10] += 1
 
    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]
 
    # Build the output array
    i = n - 1
    while i >= 0:
        index = arr[i] // exp1
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
 
    # Copying the output array to arr[],
    # so that arr now contains sorted numbers
    i = 0
    for i in range(0, len(arr)):
        arr[i] = output[i]
        plt.bar(x,arr)
        plt.pause(0.01)
        plt.clf()

 
def radixSort(arr):
 
    # Find the maximum number to know number of digits
    max1 = max(arr)
 
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp >= 1:
        radixSort1(arr,exp)
        exp *= 10
    
    plt.clf()
    plt.bar(x,arr)
    plt.show()

def bucketSort(arr, noOfBuckets):
    max_ele = max(arr)
    min_ele = min(arr)
  
    # range(for buckets)
    rnge = (max_ele - min_ele) / noOfBuckets
  
    temp = []
  
    # create empty buckets
    for i in range(noOfBuckets):
        temp.append([])
  
    # scatter the array elements
    # into the correct bucket
    for i in range(len(arr)):
        diff = (arr[i] - min_ele) / rnge - int((arr[i] - min_ele) / rnge)
        plt.bar(x,arr)
        plt.pause(0.01)
        plt.clf()
        # append the boundary elements to the lower array
        if(diff == 0 and arr[i] != min_ele):
            temp[int((arr[i] - min_ele) / rnge) - 1].append(arr[i])
  
        else:
            temp[int((arr[i] - min_ele) / rnge)].append(arr[i])
  
    # Sort each bucket individually
    for i in range(len(temp)):
        if len(temp[i]) != 0:
            temp[i].sort()
  
    # Gather sorted elements 
    # to the original array
    k = 0
    for lst in temp:
        if lst:
            for i in lst:
                plt.bar(x,arr)
                plt.pause(0.01)
                plt.clf()                 
                arr[k] = i
                k = k+1
   
    plt.clf()
    plt.bar(x,arr)
    plt.show() 
    
def partition(array, low, high):
 
    # choose the rightmost element as pivot
    pivot = array[high]
 
    # pointer for greater element
    i = low - 1
 
    # traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:
            plt.bar(list(range(amount)),array,color=['red'])
            plt.pause(0.01)
            plt.clf()
            # If element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1
 
            # Swapping element at i with element at j
            (array[i], array[j]) = (array[j], array[i])
 
    # Swap the pivot element with the greater element specified by i
    (array[i + 1], array[high]) = (array[high], array[i + 1])
 
    # Return the position from where partition is done
    return i + 1
 
# function to perform quicksort
 
 
def quickSort(array, low, high):
    if low < high:
 
        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = partition(array, low, high)
        plt.bar(list(range(amount)),array,color=['red'])
        plt.pause(0.01)
        plt.clf()
        # Recursive call on the left of pivot
        quickSort(array, low, pi - 1)
        plt.bar(list(range(amount)),array,color=['red'])
        plt.pause(0.01)
        plt.clf()
        # Recursive call on the right of pivot
        quickSort(array, pi + 1, high)
        plt.bar(list(range(amount)),array,color=['red'])
        plt.pause(0.01)
        plt.clf()

def heapify(arr, N, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1     # left = 2*i + 1
    r = 2 * i + 2     # right = 2*i + 2
 
    # See if left child of root exists and is
    # greater than root
    if l < N and arr[largest] < arr[l]:
        largest = l
 
    # See if right child of root exists and is
    # greater than root
    if r < N and arr[largest] < arr[r]:
        largest = r
 
    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap
 
        # Heapify the root.
        plt.bar(list(range(amount)),arr,color=['red'])
        plt.pause(0.01)
        plt.clf()
        heapify(arr, N, largest)
        plt.bar(list(range(amount)),arr,color=['red'])
        plt.pause(0.01)
        plt.clf()
 
# The main function to sort an array of given size
 
 
def heapSort(arr):
    N = len(arr)
 
    # Build a maxheap.
    for i in range(N//2 - 1, -1, -1):
        plt.bar(list(range(amount)),arr,color=['red'])
        plt.pause(0.01)
        plt.clf()
        heapify(arr, N, i)
        plt.bar(list(range(amount)),arr,color=['red'])
        plt.pause(0.01)
        plt.clf()
 
    # One by one extract elements
    for i in range(N-1, 0, -1):
        plt.bar(list(range(amount)),arr,color=['red'])
        plt.pause(0.01)
        plt.clf()
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)
        plt.bar(list(range(amount)),arr,color=['red'])
        plt.pause(0.01)
        plt.clf()


print("Enter choice:")
print("1- Bubble sort")
print("2- Insertion sort")
print("3- Selection sort")
print("4- Merge sort")
print("5- Count sort")
print("6- Radix sort")
print("7- Bucket sort")
print("8- Quick sort")
print("9- Heap sort")




choice = input()
if(choice=='1'):
    bubbleSort()
elif(choice=='2'):
    insertionSort()
elif(choice=='3'):
    selectionSort()
elif(choice=='4'):
    left = 0
    right = amount
    lst = [random.randint(0,1000) for _ in range(amount)]
    merge_sort(lst,left,right)
    plt.bar(list(range(amount)),lst,color=['red'])
    plt.show()
elif(choice=='5'):
    count_sort(lst)
elif(choice=='6'):
    radixSort(lst)
elif(choice=='7'):
    bucketSort(lst,5)
elif(choice=='8'):
    quickSort(lst,0,amount - 1)
    plt.bar(list(range(amount)),lst,color=['red'])
    plt.show()
elif(choice=='9'):
    heapSort(lst)
    plt.bar(list(range(amount)),lst,color=['red'])
    plt.show()

plt.close()