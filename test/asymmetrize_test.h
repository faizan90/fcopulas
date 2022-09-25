#pragma once


void quick_sort(double *arr, long long first_index, long long last_index) {

	if (first_index >= last_index) return;

    double pivot = arr[first_index + ((last_index - first_index) / 2)];

    long long index_a = first_index - 1;

    long long index_b = last_index + 1;

    double temp;

    while (1) {

    	while(arr[++index_a] < pivot);

        while(arr[--index_b] > pivot);

        if (index_a >= index_b) break;

        temp = arr[index_a];
        arr[index_a] = arr[index_b];
        arr[index_b] = temp;
    }

    quick_sort(arr, first_index, index_b);
    quick_sort(arr, index_b + 1, last_index);
	return;
}


unsigned long long searchsorted(
		const double *arr, 
		const double value, 
		const unsigned long long arr_size) {

	// arr must be sorted
	unsigned long long first = 0, last = arr_size - 1, curr_idx;

	if (value <= arr[0]) {
		return 0;
	}

	else if (value > arr[last]) {
		return arr_size;
	}

	while (first <= last) {
		curr_idx = (unsigned long long) (0.5 * (first + last));

		if ((value > arr[curr_idx]) && (value <= arr[curr_idx + 1])) {
			return curr_idx + 1;
		}

		else if (value < arr[curr_idx]) {
			last = curr_idx - 1;
		}

		else if (value > arr[curr_idx]) {
			first = curr_idx + 1;
		}

		else {
			return curr_idx;
		}
	}
	return 0;
}