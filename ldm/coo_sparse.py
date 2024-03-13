#Author: Yaojian Chen

import tenseal as ts
import torch
import copy
import numpy as np

class COOSparseTensor:
    def __init__(self, values, indices, shape):
        #assert len(values) == len(indices), "Length of values and indices must match"
        self.values = values
        # Convert index lists to tuples in the initialization
        if isinstance(indices, torch.Tensor):
            self.indices = np.array(indices)
        else:
            self.indices = np.array([tuple(index) for index in indices])
        self.shape = shape

    def to_dense(self):
        if isinstance(self.values, ts.tensors.ckksvector.CKKSVector):
            print("Wrong, encrypted tensor can not be dense")
            exit()
        dense_tensor = self._initialize_dense_tensor(self.shape)
        for value, index in zip(self.values, self.indices):
            self._set_value(dense_tensor, index, value)
        return dense_tensor

    def merge_tensor(self, dense_tensor_ori):
        dense_tensor = copy.deepcopy(dense_tensor_ori)
        #replace the elements in the dense tensor by coo tensor
        if isinstance(dense_tensor_ori, torch.Tensor):
            for value, index in zip(self.values, self.indices):
                dense_tensor[tuple(index)] = value
            return dense_tensor
        for value, index in zip(self.values, self.indices):
            self._set_value(dense_tensor, index, value)
        return dense_tensor

    def encrypt(self, context):
        #self.values = [ts.ckks_tensor(context, [i]) for i in self.values]
        self.values = ts.ckks_vector(context, self.values)
        #pass

    def decrypt_inplace(self):
        #self.values = [i.decrypt().tolist()[0] for i in self.values]
        self.values = self.values.decrypt()

    def decrypt(self):
        #values = [i.decrypt().tolist()[0] for i in self.values]
        values = self.values.decrypt()
        return COOSparseTensor(values, self.indices, self.shape)
        #return COOSparseTensor(self.values, self.indices, self.shape)

    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            #values = {index: value for index, value in zip(self.indices, self.values)}
            #for index in self.indices:
            #    values[index] = values[index] + other[index]
            #new_indices, new_values = zip(*values.items())
            #new_indices = [list(index) for index in new_indices]
            values = other[self.indices.T]
            new_values = self.values + values
            new_indices = self.indices
        elif isinstance(other, (int, float)):
            new_values = self.values + other
        
        return COOSparseTensor(new_values, new_indices, self.shape)

    def __mul__(self, other):
        if isinstance(other, torch.Tensor):
            values = other[self.indices.T]
            new_values = self.values * values
            return COOSparseTensor(new_values, self.indices, self.shape)

        elif isinstance(other, (int, float)):
            return self._scalar_mul(other)
        else:
            raise ValueError("Unsupported operand type for mul: '{}'".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)
            
    def _scalar_mul(self, scalar):
        result_values = scalar * self.values
        # Indices remain unchanged for scalar multiplication
        return COOSparseTensor(result_values, self.indices, self.shape)

    def _initialize_dense_tensor(self, shape):
        if len(shape) == 1:
            return [0] * shape[0]
        return [self._initialize_dense_tensor(shape[1:]) for _ in range(shape[0])]

    def _set_value(self, tensor, index, value):
        for i in index[:-1]:
            tensor = tensor[i]
        tensor[index[-1]] = value

def dense_to_coo(dense_tensor, prefix=[]):
    if isinstance(dense_tensor, torch.Tensor):
        indices = dense_tensor.nonzero(as_tuple=True)
        values = dense_tensor[indices]
        indices = dense_tensor.nonzero()
        return values, indices
    if isinstance(dense_tensor[0], list):  # Check if the first element is a list (indicative of higher dimensions)
        values = []
        indices = []
        for i, sublist in enumerate(dense_tensor):
            sub_values, sub_indices = dense_to_coo(sublist, prefix=prefix + [i])
            values.extend(sub_values)
            indices.extend(sub_indices)
        return values, indices
    else:  # Base case: we are at the deepest level (actual values)
        return [val for val in dense_tensor if val != 0], [prefix + [i] for i, val in enumerate(dense_tensor) if val != 0]


def convert_dense_to_coo(dense_tensor):
    values, indices = dense_to_coo(dense_tensor)
    if isinstance(dense_tensor, torch.Tensor):
        shape = dense_tensor.shape
    else:
        shape = [len(dense_tensor)]
        temp = dense_tensor
        while isinstance(temp[0], list):
            shape.append(len(temp[0]))
            temp = temp[0]
    return COOSparseTensor(values, indices, shape)

def get_encryption_context():
    # controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)

    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    return context 

if __name__ == "__main__":
    # Example usage
    context = get_encryption_context()

    values = [1, 2, 3, 4]
    indices = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    shape = (2, 2, 2)

    tensor1 = COOSparseTensor(values, indices, shape)
    tensor2 = COOSparseTensor([5, 6], [(0, 0, 1), (1, 0, 1)], shape)
    tensor3 = torch.randn(2,2,2)
    print(tensor3)

    # Convert to dense
    dense_tensor = torch.tensor(tensor1.to_dense())
    print("Dense Tensor:", dense_tensor)

    tensor1.encrypt(context)
    # Add two tensors
    added_tensor = (tensor1 + tensor3)
    added_tensor.decrypt_inplace()
    added_tensor = added_tensor.to_dense()
    print("Added Tensor:", added_tensor)
    print("result: ", dense_tensor + tensor3)

    # Multiply two tensors
    mul_tensor = (tensor1 * tensor3)
    mul_tensor.decrypt_inplace()
    mul_tensor = mul_tensor.to_dense()
    print("Multiplied Tensor:", mul_tensor)
    print("result: ", dense_tensor * tensor3)

