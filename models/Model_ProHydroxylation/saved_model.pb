��.
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.12v2.8.0-80-g0516d4d8bce8��*
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 * 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:
 *
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
: *
dtype0
�
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_5/gamma
�
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_5/beta
�
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
: *
dtype0
�
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_5/moving_mean
�
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
: *
dtype0
�
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_5/moving_variance
�
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:  *
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
: *
dtype0
�
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_6/gamma
�
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_6/beta
�
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
: *
dtype0
�
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_6/moving_mean
�
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
�
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_6/moving_variance
�
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

: *
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
: *
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:  *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
: *
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
�
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_7/gamma
�
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_7/beta
�
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
: *
dtype0
�
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_7/moving_mean
�
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
�
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_7/moving_variance
�
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
5token_and_position_embedding_1/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*F
shared_name75token_and_position_embedding_1/embedding_2/embeddings
�
Itoken_and_position_embedding_1/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_1/embedding_2/embeddings*
_output_shapes

:*
dtype0
�
5token_and_position_embedding_1/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*F
shared_name75token_and_position_embedding_1/embedding_3/embeddings
�
Itoken_and_position_embedding_1/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_1/embedding_3/embeddings*
_output_shapes

:	*
dtype0
�
*transformer_block_1/attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*transformer_block_1/attention/query/kernel
�
>transformer_block_1/attention/query/kernel/Read/ReadVariableOpReadVariableOp*transformer_block_1/attention/query/kernel*"
_output_shapes
:*
dtype0
�
(transformer_block_1/attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(transformer_block_1/attention/query/bias
�
<transformer_block_1/attention/query/bias/Read/ReadVariableOpReadVariableOp(transformer_block_1/attention/query/bias*
_output_shapes

:*
dtype0
�
(transformer_block_1/attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(transformer_block_1/attention/key/kernel
�
<transformer_block_1/attention/key/kernel/Read/ReadVariableOpReadVariableOp(transformer_block_1/attention/key/kernel*"
_output_shapes
:*
dtype0
�
&transformer_block_1/attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&transformer_block_1/attention/key/bias
�
:transformer_block_1/attention/key/bias/Read/ReadVariableOpReadVariableOp&transformer_block_1/attention/key/bias*
_output_shapes

:*
dtype0
�
*transformer_block_1/attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*transformer_block_1/attention/value/kernel
�
>transformer_block_1/attention/value/kernel/Read/ReadVariableOpReadVariableOp*transformer_block_1/attention/value/kernel*"
_output_shapes
:*
dtype0
�
(transformer_block_1/attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(transformer_block_1/attention/value/bias
�
<transformer_block_1/attention/value/bias/Read/ReadVariableOpReadVariableOp(transformer_block_1/attention/value/bias*
_output_shapes

:*
dtype0
�
5transformer_block_1/attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75transformer_block_1/attention/attention_output/kernel
�
Itransformer_block_1/attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_1/attention/attention_output/kernel*"
_output_shapes
:*
dtype0
�
3transformer_block_1/attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53transformer_block_1/attention/attention_output/bias
�
Gtransformer_block_1/attention/attention_output/bias/Read/ReadVariableOpReadVariableOp3transformer_block_1/attention/attention_output/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
�
/transformer_block_1/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_block_1/layer_normalization_2/gamma
�
Ctransformer_block_1/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_1/layer_normalization_2/gamma*
_output_shapes
:*
dtype0
�
.transformer_block_1/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.transformer_block_1/layer_normalization_2/beta
�
Btransformer_block_1/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp.transformer_block_1/layer_normalization_2/beta*
_output_shapes
:*
dtype0
�
/transformer_block_1/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_block_1/layer_normalization_3/gamma
�
Ctransformer_block_1/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_1/layer_normalization_3/gamma*
_output_shapes
:*
dtype0
�
.transformer_block_1/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.transformer_block_1/layer_normalization_3/beta
�
Btransformer_block_1/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp.transformer_block_1/layer_normalization_3/beta*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *'
shared_nameAdam/dense_10/kernel/m
�
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:
 *
dtype0
�
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_5/gamma/m
�
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
: *
dtype0
�
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_5/beta/m
�
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
: *
dtype0
�
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_11/kernel/m
�
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:  *
dtype0
�
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_6/gamma/m
�
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
: *
dtype0
�
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_6/beta/m
�
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
: *
dtype0
�
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_12/kernel/m
�
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:  *
dtype0
�
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_4/gamma/m
�
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
: *
dtype0
�
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_4/beta/m
�
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_7/gamma/m
�
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
: *
dtype0
�
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_7/beta/m
�
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
: *
dtype0
�
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_13/kernel/m
�
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
�
<Adam/token_and_position_embedding_1/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/token_and_position_embedding_1/embedding_2/embeddings/m
�
PAdam/token_and_position_embedding_1/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_1/embedding_2/embeddings/m*
_output_shapes

:*
dtype0
�
<Adam/token_and_position_embedding_1/embedding_3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*M
shared_name><Adam/token_and_position_embedding_1/embedding_3/embeddings/m
�
PAdam/token_and_position_embedding_1/embedding_3/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_1/embedding_3/embeddings/m*
_output_shapes

:	*
dtype0
�
1Adam/transformer_block_1/attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block_1/attention/query/kernel/m
�
EAdam/transformer_block_1/attention/query/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/transformer_block_1/attention/query/kernel/m*"
_output_shapes
:*
dtype0
�
/Adam/transformer_block_1/attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/transformer_block_1/attention/query/bias/m
�
CAdam/transformer_block_1/attention/query/bias/m/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_1/attention/query/bias/m*
_output_shapes

:*
dtype0
�
/Adam/transformer_block_1/attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/transformer_block_1/attention/key/kernel/m
�
CAdam/transformer_block_1/attention/key/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_1/attention/key/kernel/m*"
_output_shapes
:*
dtype0
�
-Adam/transformer_block_1/attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-Adam/transformer_block_1/attention/key/bias/m
�
AAdam/transformer_block_1/attention/key/bias/m/Read/ReadVariableOpReadVariableOp-Adam/transformer_block_1/attention/key/bias/m*
_output_shapes

:*
dtype0
�
1Adam/transformer_block_1/attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block_1/attention/value/kernel/m
�
EAdam/transformer_block_1/attention/value/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/transformer_block_1/attention/value/kernel/m*"
_output_shapes
:*
dtype0
�
/Adam/transformer_block_1/attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/transformer_block_1/attention/value/bias/m
�
CAdam/transformer_block_1/attention/value/bias/m/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_1/attention/value/bias/m*
_output_shapes

:*
dtype0
�
<Adam/transformer_block_1/attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_block_1/attention/attention_output/kernel/m
�
PAdam/transformer_block_1/attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_1/attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
�
:Adam/transformer_block_1/attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_block_1/attention/attention_output/bias/m
�
NAdam/transformer_block_1/attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_1/attention/attention_output/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
�
6Adam/transformer_block_1/layer_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_1/layer_normalization_2/gamma/m
�
JAdam/transformer_block_1/layer_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_1/layer_normalization_2/gamma/m*
_output_shapes
:*
dtype0
�
5Adam/transformer_block_1/layer_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_block_1/layer_normalization_2/beta/m
�
IAdam/transformer_block_1/layer_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_1/layer_normalization_2/beta/m*
_output_shapes
:*
dtype0
�
6Adam/transformer_block_1/layer_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_1/layer_normalization_3/gamma/m
�
JAdam/transformer_block_1/layer_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_1/layer_normalization_3/gamma/m*
_output_shapes
:*
dtype0
�
5Adam/transformer_block_1/layer_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_block_1/layer_normalization_3/beta/m
�
IAdam/transformer_block_1/layer_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_1/layer_normalization_3/beta/m*
_output_shapes
:*
dtype0
�
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *'
shared_nameAdam/dense_10/kernel/v
�
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:
 *
dtype0
�
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_5/gamma/v
�
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
: *
dtype0
�
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_5/beta/v
�
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
: *
dtype0
�
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_11/kernel/v
�
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:  *
dtype0
�
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_6/gamma/v
�
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
: *
dtype0
�
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_6/beta/v
�
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
: *
dtype0
�
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_12/kernel/v
�
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:  *
dtype0
�
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_4/gamma/v
�
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
: *
dtype0
�
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_4/beta/v
�
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_7/gamma/v
�
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
: *
dtype0
�
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_7/beta/v
�
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
: *
dtype0
�
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_13/kernel/v
�
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0
�
<Adam/token_and_position_embedding_1/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/token_and_position_embedding_1/embedding_2/embeddings/v
�
PAdam/token_and_position_embedding_1/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_1/embedding_2/embeddings/v*
_output_shapes

:*
dtype0
�
<Adam/token_and_position_embedding_1/embedding_3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*M
shared_name><Adam/token_and_position_embedding_1/embedding_3/embeddings/v
�
PAdam/token_and_position_embedding_1/embedding_3/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_1/embedding_3/embeddings/v*
_output_shapes

:	*
dtype0
�
1Adam/transformer_block_1/attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block_1/attention/query/kernel/v
�
EAdam/transformer_block_1/attention/query/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/transformer_block_1/attention/query/kernel/v*"
_output_shapes
:*
dtype0
�
/Adam/transformer_block_1/attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/transformer_block_1/attention/query/bias/v
�
CAdam/transformer_block_1/attention/query/bias/v/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_1/attention/query/bias/v*
_output_shapes

:*
dtype0
�
/Adam/transformer_block_1/attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/transformer_block_1/attention/key/kernel/v
�
CAdam/transformer_block_1/attention/key/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_1/attention/key/kernel/v*"
_output_shapes
:*
dtype0
�
-Adam/transformer_block_1/attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-Adam/transformer_block_1/attention/key/bias/v
�
AAdam/transformer_block_1/attention/key/bias/v/Read/ReadVariableOpReadVariableOp-Adam/transformer_block_1/attention/key/bias/v*
_output_shapes

:*
dtype0
�
1Adam/transformer_block_1/attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block_1/attention/value/kernel/v
�
EAdam/transformer_block_1/attention/value/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/transformer_block_1/attention/value/kernel/v*"
_output_shapes
:*
dtype0
�
/Adam/transformer_block_1/attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/transformer_block_1/attention/value/bias/v
�
CAdam/transformer_block_1/attention/value/bias/v/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_1/attention/value/bias/v*
_output_shapes

:*
dtype0
�
<Adam/transformer_block_1/attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_block_1/attention/attention_output/kernel/v
�
PAdam/transformer_block_1/attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_1/attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
�
:Adam/transformer_block_1/attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_block_1/attention/attention_output/bias/v
�
NAdam/transformer_block_1/attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_1/attention/attention_output/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
�
6Adam/transformer_block_1/layer_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_1/layer_normalization_2/gamma/v
�
JAdam/transformer_block_1/layer_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_1/layer_normalization_2/gamma/v*
_output_shapes
:*
dtype0
�
5Adam/transformer_block_1/layer_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_block_1/layer_normalization_2/beta/v
�
IAdam/transformer_block_1/layer_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_1/layer_normalization_2/beta/v*
_output_shapes
:*
dtype0
�
6Adam/transformer_block_1/layer_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_1/layer_normalization_3/gamma/v
�
JAdam/transformer_block_1/layer_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_1/layer_normalization_3/gamma/v*
_output_shapes
:*
dtype0
�
5Adam/transformer_block_1/layer_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_block_1/layer_normalization_3/beta/v
�
IAdam/transformer_block_1/layer_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_1/layer_normalization_3/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
�
%axis
	&gamma
'beta
(moving_mean
)moving_variance
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4_random_generator
5__call__
*6&call_and_return_all_conditional_losses* 
�
7	token_emb
8pos_emb
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
�

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
�
Gatt
Hffn
I
layernorm1
J
layernorm2
Kdropout1
Ldropout2
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses*
�
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h_random_generator
i__call__
*j&call_and_return_all_conditional_losses* 
�

kkernel
lbias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses*
�

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses*
�
{axis
	|gamma
}beta
~moving_mean
moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�beta_1
�beta_2

�decay
�learning_rate
	�iterm�m�&m�'m�?m�@m�Tm�Um�km�lm�sm�tm�|m�}m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�v�v�&v�'v�?v�@v�Tv�Uv�kv�lv�sv�tv�|v�}v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*
�
0
1
&2
'3
(4
)5
�6
�7
?8
@9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
T26
U27
V28
W29
k30
l31
s32
t33
|34
}35
~36
37
�38
�39
�40
�41
�42
�43*
�
0
1
&2
'3
�4
�5
?6
@7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
T24
U25
k26
l27
s28
t29
|30
}31
�32
�33
�34
�35*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

�serving_default* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
&0
'1
(2
)3*

&0
'1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 
* 
* 
* 
�
�
embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�
embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
�
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
	�axis

�gamma
	�beta
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
	�axis

�gamma
	�beta
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
T0
U1
V2
W3*

T0
U1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

k0
l1*

k0
l1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

s0
t1*

s0
t1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
|0
}1
~2
3*

|0
}1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_13/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_13/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5token_and_position_embedding_1/embedding_2/embeddings&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5token_and_position_embedding_1/embedding_3/embeddings&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*transformer_block_1/attention/query/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(transformer_block_1/attention/query/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(transformer_block_1/attention/key/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&transformer_block_1/attention/key/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*transformer_block_1/attention/value/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(transformer_block_1/attention/value/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5transformer_block_1/attention/attention_output/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3transformer_block_1/attention/attention_output/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_7/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_7/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_8/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_8/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_block_1/layer_normalization_2/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.transformer_block_1/layer_normalization_2/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_block_1/layer_normalization_3/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.transformer_block_1/layer_normalization_3/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
>
(0
)1
V2
W3
~4
5
�6
�7*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18*

�0*
* 
* 
* 
* 
* 
* 
* 
* 

(0
)1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

70
81*
* 
* 
* 
* 
* 
* 
* 
* 
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
.
G0
H1
I2
J3
K4
L5*
* 
* 
* 

V0
W1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

~0
1*
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

�total

�count
�	variables
�	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_13/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_13/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/token_and_position_embedding_1/embedding_2/embeddings/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/token_and_position_embedding_1/embedding_3/embeddings/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/transformer_block_1/attention/query/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_1/attention/query/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_1/attention/key/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/transformer_block_1/attention/key/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/transformer_block_1/attention/value/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_1/attention/value/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/transformer_block_1/attention/attention_output/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/transformer_block_1/attention/attention_output/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_7/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_7/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_8/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_8/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_block_1/layer_normalization_2/gamma/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/transformer_block_1/layer_normalization_2/beta/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_block_1/layer_normalization_3/gamma/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/transformer_block_1/layer_normalization_3/beta/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_13/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_13/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/token_and_position_embedding_1/embedding_2/embeddings/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/token_and_position_embedding_1/embedding_3/embeddings/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/transformer_block_1/attention/query/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_1/attention/query/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_1/attention/key/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/transformer_block_1/attention/key/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/transformer_block_1/attention/value/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_1/attention/value/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/transformer_block_1/attention/attention_output/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/transformer_block_1/attention/attention_output/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_7/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_7/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_8/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_8/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_block_1/layer_normalization_2/gamma/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/transformer_block_1/layer_normalization_2/beta/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_block_1/layer_normalization_3/gamma/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/transformer_block_1/layer_normalization_3/beta/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_3Placeholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
z
serving_default_input_4Placeholder*'
_output_shapes
:���������
*
dtype0*
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4dense_10/kerneldense_10/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betadense_11/kerneldense_11/bias5token_and_position_embedding_1/embedding_3/embeddings5token_and_position_embedding_1/embedding_2/embeddings%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/beta*transformer_block_1/attention/query/kernel(transformer_block_1/attention/query/bias(transformer_block_1/attention/key/kernel&transformer_block_1/attention/key/bias*transformer_block_1/attention/value/kernel(transformer_block_1/attention/value/bias5transformer_block_1/attention/attention_output/kernel3transformer_block_1/attention/attention_output/bias/transformer_block_1/layer_normalization_2/gamma.transformer_block_1/layer_normalization_2/betadense_7/kerneldense_7/biasdense_8/kerneldense_8/bias/transformer_block_1/layer_normalization_3/gamma.transformer_block_1/layer_normalization_3/betadense_12/kerneldense_12/biasdense_9/kerneldense_9/bias%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/beta%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betadense_13/kerneldense_13/bias*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_12159572
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�7
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpItoken_and_position_embedding_1/embedding_2/embeddings/Read/ReadVariableOpItoken_and_position_embedding_1/embedding_3/embeddings/Read/ReadVariableOp>transformer_block_1/attention/query/kernel/Read/ReadVariableOp<transformer_block_1/attention/query/bias/Read/ReadVariableOp<transformer_block_1/attention/key/kernel/Read/ReadVariableOp:transformer_block_1/attention/key/bias/Read/ReadVariableOp>transformer_block_1/attention/value/kernel/Read/ReadVariableOp<transformer_block_1/attention/value/bias/Read/ReadVariableOpItransformer_block_1/attention/attention_output/kernel/Read/ReadVariableOpGtransformer_block_1/attention/attention_output/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpCtransformer_block_1/layer_normalization_2/gamma/Read/ReadVariableOpBtransformer_block_1/layer_normalization_2/beta/Read/ReadVariableOpCtransformer_block_1/layer_normalization_3/gamma/Read/ReadVariableOpBtransformer_block_1/layer_normalization_3/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOpPAdam/token_and_position_embedding_1/embedding_2/embeddings/m/Read/ReadVariableOpPAdam/token_and_position_embedding_1/embedding_3/embeddings/m/Read/ReadVariableOpEAdam/transformer_block_1/attention/query/kernel/m/Read/ReadVariableOpCAdam/transformer_block_1/attention/query/bias/m/Read/ReadVariableOpCAdam/transformer_block_1/attention/key/kernel/m/Read/ReadVariableOpAAdam/transformer_block_1/attention/key/bias/m/Read/ReadVariableOpEAdam/transformer_block_1/attention/value/kernel/m/Read/ReadVariableOpCAdam/transformer_block_1/attention/value/bias/m/Read/ReadVariableOpPAdam/transformer_block_1/attention/attention_output/kernel/m/Read/ReadVariableOpNAdam/transformer_block_1/attention/attention_output/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOpJAdam/transformer_block_1/layer_normalization_2/gamma/m/Read/ReadVariableOpIAdam/transformer_block_1/layer_normalization_2/beta/m/Read/ReadVariableOpJAdam/transformer_block_1/layer_normalization_3/gamma/m/Read/ReadVariableOpIAdam/transformer_block_1/layer_normalization_3/beta/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpPAdam/token_and_position_embedding_1/embedding_2/embeddings/v/Read/ReadVariableOpPAdam/token_and_position_embedding_1/embedding_3/embeddings/v/Read/ReadVariableOpEAdam/transformer_block_1/attention/query/kernel/v/Read/ReadVariableOpCAdam/transformer_block_1/attention/query/bias/v/Read/ReadVariableOpCAdam/transformer_block_1/attention/key/kernel/v/Read/ReadVariableOpAAdam/transformer_block_1/attention/key/bias/v/Read/ReadVariableOpEAdam/transformer_block_1/attention/value/kernel/v/Read/ReadVariableOpCAdam/transformer_block_1/attention/value/bias/v/Read/ReadVariableOpPAdam/transformer_block_1/attention/attention_output/kernel/v/Read/ReadVariableOpNAdam/transformer_block_1/attention/attention_output/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOpJAdam/transformer_block_1/layer_normalization_2/gamma/v/Read/ReadVariableOpIAdam/transformer_block_1/layer_normalization_2/beta/v/Read/ReadVariableOpJAdam/transformer_block_1/layer_normalization_3/gamma/v/Read/ReadVariableOpIAdam/transformer_block_1/layer_normalization_3/beta/v/Read/ReadVariableOpConst*�
Tin�
2}	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_12161111
�#
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense_11/kerneldense_11/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancedense_9/kerneldense_9/biasdense_12/kerneldense_12/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_13/kerneldense_13/biasbeta_1beta_2decaylearning_rate	Adam/iter5token_and_position_embedding_1/embedding_2/embeddings5token_and_position_embedding_1/embedding_3/embeddings*transformer_block_1/attention/query/kernel(transformer_block_1/attention/query/bias(transformer_block_1/attention/key/kernel&transformer_block_1/attention/key/bias*transformer_block_1/attention/value/kernel(transformer_block_1/attention/value/bias5transformer_block_1/attention/attention_output/kernel3transformer_block_1/attention/attention_output/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias/transformer_block_1/layer_normalization_2/gamma.transformer_block_1/layer_normalization_2/beta/transformer_block_1/layer_normalization_3/gamma.transformer_block_1/layer_normalization_3/betatotalcountAdam/dense_10/kernel/mAdam/dense_10/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/dense_11/kernel/mAdam/dense_11/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/dense_13/kernel/mAdam/dense_13/bias/m<Adam/token_and_position_embedding_1/embedding_2/embeddings/m<Adam/token_and_position_embedding_1/embedding_3/embeddings/m1Adam/transformer_block_1/attention/query/kernel/m/Adam/transformer_block_1/attention/query/bias/m/Adam/transformer_block_1/attention/key/kernel/m-Adam/transformer_block_1/attention/key/bias/m1Adam/transformer_block_1/attention/value/kernel/m/Adam/transformer_block_1/attention/value/bias/m<Adam/transformer_block_1/attention/attention_output/kernel/m:Adam/transformer_block_1/attention/attention_output/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/m6Adam/transformer_block_1/layer_normalization_2/gamma/m5Adam/transformer_block_1/layer_normalization_2/beta/m6Adam/transformer_block_1/layer_normalization_3/gamma/m5Adam/transformer_block_1/layer_normalization_3/beta/mAdam/dense_10/kernel/vAdam/dense_10/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/dense_11/kernel/vAdam/dense_11/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/vAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/dense_13/kernel/vAdam/dense_13/bias/v<Adam/token_and_position_embedding_1/embedding_2/embeddings/v<Adam/token_and_position_embedding_1/embedding_3/embeddings/v1Adam/transformer_block_1/attention/query/kernel/v/Adam/transformer_block_1/attention/query/bias/v/Adam/transformer_block_1/attention/key/kernel/v-Adam/transformer_block_1/attention/key/bias/v1Adam/transformer_block_1/attention/value/kernel/v/Adam/transformer_block_1/attention/value/bias/v<Adam/transformer_block_1/attention/attention_output/kernel/v:Adam/transformer_block_1/attention/attention_output/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/v6Adam/transformer_block_1/layer_normalization_2/gamma/v5Adam/transformer_block_1/layer_normalization_2/beta/v6Adam/transformer_block_1/layer_normalization_3/gamma/v5Adam/transformer_block_1/layer_normalization_3/beta/v*�
Tin�
~2|*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_12161490��$
�`
�
E__inference_model_1_layer_call_and_return_conditional_losses_12157596

inputs
inputs_1#
dense_10_12157256:
 
dense_10_12157258: ,
batch_normalization_5_12157261: ,
batch_normalization_5_12157263: ,
batch_normalization_5_12157265: ,
batch_normalization_5_12157267: #
dense_11_12157289:  
dense_11_12157291: 9
'token_and_position_embedding_1_12157320:	9
'token_and_position_embedding_1_12157322:,
batch_normalization_6_12157325: ,
batch_normalization_6_12157327: ,
batch_normalization_6_12157329: ,
batch_normalization_6_12157331: 2
transformer_block_1_12157462:.
transformer_block_1_12157464:2
transformer_block_1_12157466:.
transformer_block_1_12157468:2
transformer_block_1_12157470:.
transformer_block_1_12157472:2
transformer_block_1_12157474:*
transformer_block_1_12157476:*
transformer_block_1_12157478:*
transformer_block_1_12157480:.
transformer_block_1_12157482:*
transformer_block_1_12157484:.
transformer_block_1_12157486:*
transformer_block_1_12157488:*
transformer_block_1_12157490:*
transformer_block_1_12157492:#
dense_12_12157515:  
dense_12_12157517: "
dense_9_12157532: 
dense_9_12157534: ,
batch_normalization_7_12157537: ,
batch_normalization_7_12157539: ,
batch_normalization_7_12157541: ,
batch_normalization_7_12157543: ,
batch_normalization_4_12157546: ,
batch_normalization_4_12157548: ,
batch_normalization_4_12157550: ,
batch_normalization_4_12157552: #
dense_13_12157590:@
dense_13_12157592:
identity��-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�6token_and_position_embedding_1/StatefulPartitionedCall�+transformer_block_1/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_10_12157256dense_10_12157258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_12157255�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_12157261batch_normalization_5_12157263batch_normalization_5_12157265batch_normalization_5_12157267*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12156725�
dropout_9/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12157275�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_11_12157289dense_11_12157291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_12157288�
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs'token_and_position_embedding_1_12157320'token_and_position_embedding_1_12157322*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *e
f`R^
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12157319�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_6_12157325batch_normalization_6_12157327batch_normalization_6_12157329batch_normalization_6_12157331*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12157000�
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0transformer_block_1_12157462transformer_block_1_12157464transformer_block_1_12157466transformer_block_1_12157468transformer_block_1_12157470transformer_block_1_12157472transformer_block_1_12157474transformer_block_1_12157476transformer_block_1_12157478transformer_block_1_12157480transformer_block_1_12157482transformer_block_1_12157484transformer_block_1_12157486transformer_block_1_12157488transformer_block_1_12157490transformer_block_1_12157492*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12157461�
dropout_10/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12157500�
*global_average_pooling1d_1/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12157068�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_12_12157515dense_12_12157517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_12157514�
dense_9/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_9_12157532dense_9_12157534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_12157531�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_7_12157537batch_normalization_7_12157539batch_normalization_7_12157541batch_normalization_7_12157543*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12157177�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_12157546batch_normalization_4_12157548batch_normalization_4_12157550batch_normalization_4_12157552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12157095�
dropout_8/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_12157560�
dropout_11/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12157567�
concatenate_1/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0#dropout_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12157576�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_13_12157590dense_13_12157592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_12157589x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������

 
_user_specified_nameinputs
�f
�
E__inference_model_1_layer_call_and_return_conditional_losses_12158268

inputs
inputs_1#
dense_10_12158162:
 
dense_10_12158164: ,
batch_normalization_5_12158167: ,
batch_normalization_5_12158169: ,
batch_normalization_5_12158171: ,
batch_normalization_5_12158173: #
dense_11_12158177:  
dense_11_12158179: 9
'token_and_position_embedding_1_12158182:	9
'token_and_position_embedding_1_12158184:,
batch_normalization_6_12158187: ,
batch_normalization_6_12158189: ,
batch_normalization_6_12158191: ,
batch_normalization_6_12158193: 2
transformer_block_1_12158196:.
transformer_block_1_12158198:2
transformer_block_1_12158200:.
transformer_block_1_12158202:2
transformer_block_1_12158204:.
transformer_block_1_12158206:2
transformer_block_1_12158208:*
transformer_block_1_12158210:*
transformer_block_1_12158212:*
transformer_block_1_12158214:.
transformer_block_1_12158216:*
transformer_block_1_12158218:.
transformer_block_1_12158220:*
transformer_block_1_12158222:*
transformer_block_1_12158224:*
transformer_block_1_12158226:#
dense_12_12158231:  
dense_12_12158233: "
dense_9_12158236: 
dense_9_12158238: ,
batch_normalization_7_12158241: ,
batch_normalization_7_12158243: ,
batch_normalization_7_12158245: ,
batch_normalization_7_12158247: ,
batch_normalization_4_12158250: ,
batch_normalization_4_12158252: ,
batch_normalization_4_12158254: ,
batch_normalization_4_12158256: #
dense_13_12158262:@
dense_13_12158264:
identity��-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�"dropout_10/StatefulPartitionedCall�"dropout_11/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�6token_and_position_embedding_1/StatefulPartitionedCall�+transformer_block_1/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_10_12158162dense_10_12158164*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_12157255�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_12158167batch_normalization_5_12158169batch_normalization_5_12158171batch_normalization_5_12158173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12156772�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12158048�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_11_12158177dense_11_12158179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_12157288�
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs'token_and_position_embedding_1_12158182'token_and_position_embedding_1_12158184*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *e
f`R^
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12157319�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_6_12158187batch_normalization_6_12158189batch_normalization_6_12158191batch_normalization_6_12158193*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12157047�
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0transformer_block_1_12158196transformer_block_1_12158198transformer_block_1_12158200transformer_block_1_12158202transformer_block_1_12158204transformer_block_1_12158206transformer_block_1_12158208transformer_block_1_12158210transformer_block_1_12158212transformer_block_1_12158214transformer_block_1_12158216transformer_block_1_12158218transformer_block_1_12158220transformer_block_1_12158222transformer_block_1_12158224transformer_block_1_12158226*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12157973�
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12157790�
*global_average_pooling1d_1/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12157068�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_12_12158231dense_12_12158233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_12157514�
dense_9/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_9_12158236dense_9_12158238*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_12157531�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_7_12158241batch_normalization_7_12158243batch_normalization_7_12158245batch_normalization_7_12158247*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12157224�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_12158250batch_normalization_4_12158252batch_normalization_4_12158254batch_normalization_4_12158256*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12157142�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_12157747�
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12157724�
concatenate_1/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0+dropout_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12157576�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_13_12158262dense_13_12158264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_12157589x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
F__inference_dense_10_layer_call_and_return_conditional_losses_12157255

inputs0
matmul_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
g
H__inference_dropout_10_layer_call_and_return_conditional_losses_12160212

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
t
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12160185

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

�
F__inference_dense_11_layer_call_and_return_conditional_losses_12157288

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
6__inference_transformer_block_1_layer_call_fn_12159790

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12157461s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������	: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
E__inference_dense_9_layer_call_and_return_conditional_losses_12157531

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_1_layer_call_fn_12156948
dense_7_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156924s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������	
'
_user_specified_namedense_7_input
�
f
H__inference_dropout_10_layer_call_and_return_conditional_losses_12160200

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�0
E__inference_model_1_layer_call_and_return_conditional_losses_12159476
inputs_0
inputs_19
'dense_10_matmul_readvariableop_resource:
 6
(dense_10_biasadd_readvariableop_resource: K
=batch_normalization_5_assignmovingavg_readvariableop_resource: M
?batch_normalization_5_assignmovingavg_1_readvariableop_resource: I
;batch_normalization_5_batchnorm_mul_readvariableop_resource: E
7batch_normalization_5_batchnorm_readvariableop_resource: 9
'dense_11_matmul_readvariableop_resource:  6
(dense_11_biasadd_readvariableop_resource: V
Dtoken_and_position_embedding_1_embedding_3_embedding_lookup_12159188:	V
Dtoken_and_position_embedding_1_embedding_2_embedding_lookup_12159194:K
=batch_normalization_6_assignmovingavg_readvariableop_resource: M
?batch_normalization_6_assignmovingavg_1_readvariableop_resource: I
;batch_normalization_6_batchnorm_mul_readvariableop_resource: E
7batch_normalization_6_batchnorm_readvariableop_resource: _
Itransformer_block_1_attention_query_einsum_einsum_readvariableop_resource:Q
?transformer_block_1_attention_query_add_readvariableop_resource:]
Gtransformer_block_1_attention_key_einsum_einsum_readvariableop_resource:O
=transformer_block_1_attention_key_add_readvariableop_resource:_
Itransformer_block_1_attention_value_einsum_einsum_readvariableop_resource:Q
?transformer_block_1_attention_value_add_readvariableop_resource:j
Ttransformer_block_1_attention_attention_output_einsum_einsum_readvariableop_resource:X
Jtransformer_block_1_attention_attention_output_add_readvariableop_resource:]
Otransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource:Y
Ktransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource:\
Jtransformer_block_1_sequential_1_dense_7_tensordot_readvariableop_resource:V
Htransformer_block_1_sequential_1_dense_7_biasadd_readvariableop_resource:\
Jtransformer_block_1_sequential_1_dense_8_tensordot_readvariableop_resource:V
Htransformer_block_1_sequential_1_dense_8_biasadd_readvariableop_resource:]
Otransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource:Y
Ktransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:  6
(dense_12_biasadd_readvariableop_resource: 8
&dense_9_matmul_readvariableop_resource: 5
'dense_9_biasadd_readvariableop_resource: K
=batch_normalization_7_assignmovingavg_readvariableop_resource: M
?batch_normalization_7_assignmovingavg_1_readvariableop_resource: I
;batch_normalization_7_batchnorm_mul_readvariableop_resource: E
7batch_normalization_7_batchnorm_readvariableop_resource: K
=batch_normalization_4_assignmovingavg_readvariableop_resource: M
?batch_normalization_4_assignmovingavg_1_readvariableop_resource: I
;batch_normalization_4_batchnorm_mul_readvariableop_resource: E
7batch_normalization_4_batchnorm_readvariableop_resource: 9
'dense_13_matmul_readvariableop_resource:@6
(dense_13_biasadd_readvariableop_resource:
identity��%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_4/batchnorm/ReadVariableOp�2batch_normalization_4/batchnorm/mul/ReadVariableOp�%batch_normalization_5/AssignMovingAvg�4batch_normalization_5/AssignMovingAvg/ReadVariableOp�'batch_normalization_5/AssignMovingAvg_1�6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_5/batchnorm/ReadVariableOp�2batch_normalization_5/batchnorm/mul/ReadVariableOp�%batch_normalization_6/AssignMovingAvg�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�'batch_normalization_6/AssignMovingAvg_1�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_6/batchnorm/ReadVariableOp�2batch_normalization_6/batchnorm/mul/ReadVariableOp�%batch_normalization_7/AssignMovingAvg�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�'batch_normalization_7/AssignMovingAvg_1�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_7/batchnorm/ReadVariableOp�2batch_normalization_7/batchnorm/mul/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�;token_and_position_embedding_1/embedding_2/embedding_lookup�;token_and_position_embedding_1/embedding_3/embedding_lookup�Atransformer_block_1/attention/attention_output/add/ReadVariableOp�Ktransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp�4transformer_block_1/attention/key/add/ReadVariableOp�>transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp�6transformer_block_1/attention/query/add/ReadVariableOp�@transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp�6transformer_block_1/attention/value/add/ReadVariableOp�@transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp�Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp�Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp�Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp�Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp�?transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp�Atransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp�?transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp�Atransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype0}
dense_10/MatMulMatMulinputs_1&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� ~
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_5/moments/meanMeandense_10/Relu:activations:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

: �
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
+batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
: �
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
%batch_normalization_5/AssignMovingAvgAssignSubVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: �
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: �
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� \
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_9/dropout/MulMul)batch_normalization_5/batchnorm/add_1:z:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:��������� p
dropout_9/dropout/ShapeShape)batch_normalization_5/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� �
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_11/MatMulMatMuldropout_9/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� \
$token_and_position_embedding_1/ShapeShapeinputs_0*
T0*
_output_shapes
:�
2token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������~
4token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,token_and_position_embedding_1/strided_sliceStridedSlice-token_and_position_embedding_1/Shape:output:0;token_and_position_embedding_1/strided_slice/stack:output:0=token_and_position_embedding_1/strided_slice/stack_1:output:0=token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : l
*token_and_position_embedding_1/range/limitConst*
_output_shapes
: *
dtype0*
value	B :	l
*token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
$token_and_position_embedding_1/rangeRange3token_and_position_embedding_1/range/start:output:03token_and_position_embedding_1/range/limit:output:03token_and_position_embedding_1/range/delta:output:0*
_output_shapes
:	�
;token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherDtoken_and_position_embedding_1_embedding_3_embedding_lookup_12159188-token_and_position_embedding_1/range:output:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/12159188*
_output_shapes

:	*
dtype0�
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_3/embedding_lookup:output:0*
T0*W
_classM
KIloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/12159188*
_output_shapes

:	�
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	�
/token_and_position_embedding_1/embedding_2/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������	�
;token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherDtoken_and_position_embedding_1_embedding_2_embedding_lookup_121591943token_and_position_embedding_1/embedding_2/Cast:y:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/12159194*+
_output_shapes
:���������	*
dtype0�
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_2/embedding_lookup:output:0*
T0*W
_classM
KIloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/12159194*+
_output_shapes
:���������	�
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
"token_and_position_embedding_1/addAddV2Otoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	~
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_6/moments/meanMeandense_11/Relu:activations:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes

: �
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_11/Relu:activations:03batch_normalization_6/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
+batch_normalization_6/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*
_output_shapes
: �
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
%batch_normalization_6/AssignMovingAvgAssignSubVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_6/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
'batch_normalization_6/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: �
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
%batch_normalization_6/batchnorm/mul_1Muldense_11/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: �
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
@transformer_block_1/attention/query/einsum/Einsum/ReadVariableOpReadVariableOpItransformer_block_1_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
1transformer_block_1/attention/query/einsum/EinsumEinsum&token_and_position_embedding_1/add:z:0Htransformer_block_1/attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
6transformer_block_1/attention/query/add/ReadVariableOpReadVariableOp?transformer_block_1_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
'transformer_block_1/attention/query/addAddV2:transformer_block_1/attention/query/einsum/Einsum:output:0>transformer_block_1/attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
>transformer_block_1/attention/key/einsum/Einsum/ReadVariableOpReadVariableOpGtransformer_block_1_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
/transformer_block_1/attention/key/einsum/EinsumEinsum&token_and_position_embedding_1/add:z:0Ftransformer_block_1/attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
4transformer_block_1/attention/key/add/ReadVariableOpReadVariableOp=transformer_block_1_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
%transformer_block_1/attention/key/addAddV28transformer_block_1/attention/key/einsum/Einsum:output:0<transformer_block_1/attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
@transformer_block_1/attention/value/einsum/Einsum/ReadVariableOpReadVariableOpItransformer_block_1_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
1transformer_block_1/attention/value/einsum/EinsumEinsum&token_and_position_embedding_1/add:z:0Htransformer_block_1/attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
6transformer_block_1/attention/value/add/ReadVariableOpReadVariableOp?transformer_block_1_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
'transformer_block_1/attention/value/addAddV2:transformer_block_1/attention/value/einsum/Einsum:output:0>transformer_block_1/attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	h
#transformer_block_1/attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
!transformer_block_1/attention/MulMul+transformer_block_1/attention/query/add:z:0,transformer_block_1/attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
+transformer_block_1/attention/einsum/EinsumEinsum)transformer_block_1/attention/key/add:z:0%transformer_block_1/attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
-transformer_block_1/attention/softmax/SoftmaxSoftmax4transformer_block_1/attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
-transformer_block_1/attention/einsum_1/EinsumEinsum7transformer_block_1/attention/softmax/Softmax:softmax:0+transformer_block_1/attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
Ktransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_1_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
<transformer_block_1/attention/attention_output/einsum/EinsumEinsum6transformer_block_1/attention/einsum_1/Einsum:output:0Stransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
Atransformer_block_1/attention/attention_output/add/ReadVariableOpReadVariableOpJtransformer_block_1_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
2transformer_block_1/attention/attention_output/addAddV2Etransformer_block_1/attention/attention_output/einsum/Einsum:output:0Itransformer_block_1/attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	p
+transformer_block_1/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
)transformer_block_1/dropout_6/dropout/MulMul6transformer_block_1/attention/attention_output/add:z:04transformer_block_1/dropout_6/dropout/Const:output:0*
T0*+
_output_shapes
:���������	�
+transformer_block_1/dropout_6/dropout/ShapeShape6transformer_block_1/attention/attention_output/add:z:0*
T0*
_output_shapes
:�
Btransformer_block_1/dropout_6/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_1/dropout_6/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0y
4transformer_block_1/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
2transformer_block_1/dropout_6/dropout/GreaterEqualGreaterEqualKtransformer_block_1/dropout_6/dropout/random_uniform/RandomUniform:output:0=transformer_block_1/dropout_6/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
*transformer_block_1/dropout_6/dropout/CastCast6transformer_block_1/dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
+transformer_block_1/dropout_6/dropout/Mul_1Mul-transformer_block_1/dropout_6/dropout/Mul:z:0.transformer_block_1/dropout_6/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	�
transformer_block_1/addAddV2&token_and_position_embedding_1/add:z:0/transformer_block_1/dropout_6/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	�
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block_1/layer_normalization_2/moments/meanMeantransformer_block_1/add:z:0Qtransformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
>transformer_block_1/layer_normalization_2/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add:z:0Gtransformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block_1/layer_normalization_2/moments/varianceMeanGtransformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(~
9transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block_1/layer_normalization_2/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_2/moments/variance:output:0Btransformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
7transformer_block_1/layer_normalization_2/batchnorm/mulMul=transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_2/batchnorm/mul_1Multransformer_block_1/add:z:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_2/moments/mean:output:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
7transformer_block_1/layer_normalization_2/batchnorm/subSubJtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
Atransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
7transformer_block_1/sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_block_1/sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8transformer_block_1/sequential_1/dense_7/Tensordot/ShapeShape=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
@transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_7/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_7/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Btransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_7/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_7/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8transformer_block_1/sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block_1/sequential_1/dense_7/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:transformer_block_1/sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9transformer_block_1/sequential_1/dense_7/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>transformer_block_1/sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block_1/sequential_1/dense_7/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_7/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_7/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8transformer_block_1/sequential_1/dense_7/Tensordot/stackPack@transformer_block_1/sequential_1/dense_7/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<transformer_block_1/sequential_1/dense_7/Tensordot/transpose	Transpose=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Btransformer_block_1/sequential_1/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
:transformer_block_1/sequential_1/dense_7/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_7/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9transformer_block_1/sequential_1/dense_7/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_7/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
:transformer_block_1/sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
@transformer_block_1/sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block_1/sequential_1/dense_7/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_7/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2transformer_block_1/sequential_1/dense_7/TensordotReshapeCtransformer_block_1/sequential_1/dense_7/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
?transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
0transformer_block_1/sequential_1/dense_7/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_7/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
-transformer_block_1/sequential_1/dense_7/ReluRelu9transformer_block_1/sequential_1/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
Atransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
7transformer_block_1/sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_block_1/sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8transformer_block_1/sequential_1/dense_8/Tensordot/ShapeShape;transformer_block_1/sequential_1/dense_7/Relu:activations:0*
T0*
_output_shapes
:�
@transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_8/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_8/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Btransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_8/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_8/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8transformer_block_1/sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block_1/sequential_1/dense_8/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:transformer_block_1/sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9transformer_block_1/sequential_1/dense_8/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>transformer_block_1/sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block_1/sequential_1/dense_8/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_8/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_8/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8transformer_block_1/sequential_1/dense_8/Tensordot/stackPack@transformer_block_1/sequential_1/dense_8/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<transformer_block_1/sequential_1/dense_8/Tensordot/transpose	Transpose;transformer_block_1/sequential_1/dense_7/Relu:activations:0Btransformer_block_1/sequential_1/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
:transformer_block_1/sequential_1/dense_8/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_8/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9transformer_block_1/sequential_1/dense_8/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_8/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
:transformer_block_1/sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
@transformer_block_1/sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block_1/sequential_1/dense_8/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_8/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2transformer_block_1/sequential_1/dense_8/TensordotReshapeCtransformer_block_1/sequential_1/dense_8/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
?transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
0transformer_block_1/sequential_1/dense_8/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_8/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	p
+transformer_block_1/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
)transformer_block_1/dropout_7/dropout/MulMul9transformer_block_1/sequential_1/dense_8/BiasAdd:output:04transformer_block_1/dropout_7/dropout/Const:output:0*
T0*+
_output_shapes
:���������	�
+transformer_block_1/dropout_7/dropout/ShapeShape9transformer_block_1/sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:�
Btransformer_block_1/dropout_7/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_1/dropout_7/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0y
4transformer_block_1/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
2transformer_block_1/dropout_7/dropout/GreaterEqualGreaterEqualKtransformer_block_1/dropout_7/dropout/random_uniform/RandomUniform:output:0=transformer_block_1/dropout_7/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
*transformer_block_1/dropout_7/dropout/CastCast6transformer_block_1/dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
+transformer_block_1/dropout_7/dropout/Mul_1Mul-transformer_block_1/dropout_7/dropout/Mul:z:0.transformer_block_1/dropout_7/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	�
transformer_block_1/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0/transformer_block_1/dropout_7/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	�
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block_1/layer_normalization_3/moments/meanMeantransformer_block_1/add_1:z:0Qtransformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
>transformer_block_1/layer_normalization_3/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add_1:z:0Gtransformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block_1/layer_normalization_3/moments/varianceMeanGtransformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(~
9transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block_1/layer_normalization_3/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_3/moments/variance:output:0Btransformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
7transformer_block_1/layer_normalization_3/batchnorm/mulMul=transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_3/batchnorm/mul_1Multransformer_block_1/add_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_3/moments/mean:output:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
7transformer_block_1/layer_normalization_3/batchnorm/subSubJtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout/MulMul)batch_normalization_6/batchnorm/add_1:z:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:��������� q
dropout_10/dropout/ShapeShape)batch_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� s
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d_1/MeanMean=transformer_block_1/layer_normalization_3/batchnorm/add_1:z:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_12/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_9/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� ~
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_7/moments/meanMeandense_12/Relu:activations:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes

: �
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_12/Relu:activations:03batch_normalization_7/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
+batch_normalization_7/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*
_output_shapes
: �
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
%batch_normalization_7/AssignMovingAvgAssignSubVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_7/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
'batch_normalization_7/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: �
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
%batch_normalization_7/batchnorm/mul_1Muldense_12/Relu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: �
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� ~
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_4/moments/meanMeandense_9/Relu:activations:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

: �
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_9/Relu:activations:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
: �
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: �
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: �
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� \
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_8/dropout/MulMul)batch_normalization_4/batchnorm/add_1:z:0 dropout_8/dropout/Const:output:0*
T0*'
_output_shapes
:��������� p
dropout_8/dropout/ShapeShape)batch_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� ]
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_11/dropout/MulMul)batch_normalization_7/batchnorm/add_1:z:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:��������� q
dropout_11/dropout/ShapeShape)batch_normalization_7/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0f
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� [
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2dropout_8/dropout/Mul_1:z:0dropout_11/dropout/Mul_1:z:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_13/MatMulMatMulconcatenate_1/concat:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp&^batch_normalization_6/AssignMovingAvg5^batch_normalization_6/AssignMovingAvg/ReadVariableOp(^batch_normalization_6/AssignMovingAvg_17^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_6/batchnorm/ReadVariableOp3^batch_normalization_6/batchnorm/mul/ReadVariableOp&^batch_normalization_7/AssignMovingAvg5^batch_normalization_7/AssignMovingAvg/ReadVariableOp(^batch_normalization_7/AssignMovingAvg_17^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp3^batch_normalization_7/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp<^token_and_position_embedding_1/embedding_2/embedding_lookup<^token_and_position_embedding_1/embedding_3/embedding_lookupB^transformer_block_1/attention/attention_output/add/ReadVariableOpL^transformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp5^transformer_block_1/attention/key/add/ReadVariableOp?^transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp7^transformer_block_1/attention/query/add/ReadVariableOpA^transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp7^transformer_block_1/attention/value/add/ReadVariableOpA^transformer_block_1/attention/value/einsum/Einsum/ReadVariableOpC^transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpC^transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp@^transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp@^transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2N
%batch_normalization_5/AssignMovingAvg%batch_normalization_5/AssignMovingAvg2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_1'batch_normalization_5/AssignMovingAvg_12p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2N
%batch_normalization_6/AssignMovingAvg%batch_normalization_6/AssignMovingAvg2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_6/AssignMovingAvg_1'batch_normalization_6/AssignMovingAvg_12p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2N
%batch_normalization_7/AssignMovingAvg%batch_normalization_7/AssignMovingAvg2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_7/AssignMovingAvg_1'batch_normalization_7/AssignMovingAvg_12p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2z
;token_and_position_embedding_1/embedding_2/embedding_lookup;token_and_position_embedding_1/embedding_2/embedding_lookup2z
;token_and_position_embedding_1/embedding_3/embedding_lookup;token_and_position_embedding_1/embedding_3/embedding_lookup2�
Atransformer_block_1/attention/attention_output/add/ReadVariableOpAtransformer_block_1/attention/attention_output/add/ReadVariableOp2�
Ktransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOpKtransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp2l
4transformer_block_1/attention/key/add/ReadVariableOp4transformer_block_1/attention/key/add/ReadVariableOp2�
>transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp>transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp2p
6transformer_block_1/attention/query/add/ReadVariableOp6transformer_block_1/attention/query/add/ReadVariableOp2�
@transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp@transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp2p
6transformer_block_1/attention/value/add/ReadVariableOp6transformer_block_1/attention/value/add/ReadVariableOp2�
@transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp@transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp2�
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2�
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2�
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2�
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2�
?transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp2�
Atransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp2�
?transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp2�
Atransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs/1
�
�
A__inference_token_and_position_embedding_1_layer_call_fn_12159708
x
unknown:	
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *e
f`R^
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12157319s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������	

_user_specified_namex
�
w
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12160479
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:��������� :��������� :Q M
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
�
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12159638

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
f
H__inference_dropout_11_layer_call_and_return_conditional_losses_12157567

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_1_layer_call_fn_12160525

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156924s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_6_layer_call_fn_12160107

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12157000o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_10_layer_call_and_return_conditional_losses_12159592

inputs0
matmul_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_5_layer_call_fn_12159605

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12156725o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_7_layer_call_fn_12160358

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12157224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
e
,__inference_dropout_9_layer_call_fn_12159682

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12158048o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_13_layer_call_and_return_conditional_losses_12160499

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
\
0__inference_concatenate_1_layer_call_fn_12160472
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12157576`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:��������� :��������� :Q M
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
��
�-
E__inference_model_1_layer_call_and_return_conditional_losses_12159123
inputs_0
inputs_19
'dense_10_matmul_readvariableop_resource:
 6
(dense_10_biasadd_readvariableop_resource: E
7batch_normalization_5_batchnorm_readvariableop_resource: I
;batch_normalization_5_batchnorm_mul_readvariableop_resource: G
9batch_normalization_5_batchnorm_readvariableop_1_resource: G
9batch_normalization_5_batchnorm_readvariableop_2_resource: 9
'dense_11_matmul_readvariableop_resource:  6
(dense_11_biasadd_readvariableop_resource: V
Dtoken_and_position_embedding_1_embedding_3_embedding_lookup_12158911:	V
Dtoken_and_position_embedding_1_embedding_2_embedding_lookup_12158917:E
7batch_normalization_6_batchnorm_readvariableop_resource: I
;batch_normalization_6_batchnorm_mul_readvariableop_resource: G
9batch_normalization_6_batchnorm_readvariableop_1_resource: G
9batch_normalization_6_batchnorm_readvariableop_2_resource: _
Itransformer_block_1_attention_query_einsum_einsum_readvariableop_resource:Q
?transformer_block_1_attention_query_add_readvariableop_resource:]
Gtransformer_block_1_attention_key_einsum_einsum_readvariableop_resource:O
=transformer_block_1_attention_key_add_readvariableop_resource:_
Itransformer_block_1_attention_value_einsum_einsum_readvariableop_resource:Q
?transformer_block_1_attention_value_add_readvariableop_resource:j
Ttransformer_block_1_attention_attention_output_einsum_einsum_readvariableop_resource:X
Jtransformer_block_1_attention_attention_output_add_readvariableop_resource:]
Otransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource:Y
Ktransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource:\
Jtransformer_block_1_sequential_1_dense_7_tensordot_readvariableop_resource:V
Htransformer_block_1_sequential_1_dense_7_biasadd_readvariableop_resource:\
Jtransformer_block_1_sequential_1_dense_8_tensordot_readvariableop_resource:V
Htransformer_block_1_sequential_1_dense_8_biasadd_readvariableop_resource:]
Otransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource:Y
Ktransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:  6
(dense_12_biasadd_readvariableop_resource: 8
&dense_9_matmul_readvariableop_resource: 5
'dense_9_biasadd_readvariableop_resource: E
7batch_normalization_7_batchnorm_readvariableop_resource: I
;batch_normalization_7_batchnorm_mul_readvariableop_resource: G
9batch_normalization_7_batchnorm_readvariableop_1_resource: G
9batch_normalization_7_batchnorm_readvariableop_2_resource: E
7batch_normalization_4_batchnorm_readvariableop_resource: I
;batch_normalization_4_batchnorm_mul_readvariableop_resource: G
9batch_normalization_4_batchnorm_readvariableop_1_resource: G
9batch_normalization_4_batchnorm_readvariableop_2_resource: 9
'dense_13_matmul_readvariableop_resource:@6
(dense_13_biasadd_readvariableop_resource:
identity��.batch_normalization_4/batchnorm/ReadVariableOp�0batch_normalization_4/batchnorm/ReadVariableOp_1�0batch_normalization_4/batchnorm/ReadVariableOp_2�2batch_normalization_4/batchnorm/mul/ReadVariableOp�.batch_normalization_5/batchnorm/ReadVariableOp�0batch_normalization_5/batchnorm/ReadVariableOp_1�0batch_normalization_5/batchnorm/ReadVariableOp_2�2batch_normalization_5/batchnorm/mul/ReadVariableOp�.batch_normalization_6/batchnorm/ReadVariableOp�0batch_normalization_6/batchnorm/ReadVariableOp_1�0batch_normalization_6/batchnorm/ReadVariableOp_2�2batch_normalization_6/batchnorm/mul/ReadVariableOp�.batch_normalization_7/batchnorm/ReadVariableOp�0batch_normalization_7/batchnorm/ReadVariableOp_1�0batch_normalization_7/batchnorm/ReadVariableOp_2�2batch_normalization_7/batchnorm/mul/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�;token_and_position_embedding_1/embedding_2/embedding_lookup�;token_and_position_embedding_1/embedding_3/embedding_lookup�Atransformer_block_1/attention/attention_output/add/ReadVariableOp�Ktransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp�4transformer_block_1/attention/key/add/ReadVariableOp�>transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp�6transformer_block_1/attention/query/add/ReadVariableOp�@transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp�6transformer_block_1/attention/value/add/ReadVariableOp�@transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp�Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp�Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp�Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp�Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp�?transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp�Atransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp�?transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp�Atransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype0}
dense_10/MatMulMatMulinputs_1&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: �
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: �
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� {
dropout_9/IdentityIdentity)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� �
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_11/MatMulMatMuldropout_9/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� \
$token_and_position_embedding_1/ShapeShapeinputs_0*
T0*
_output_shapes
:�
2token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������~
4token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,token_and_position_embedding_1/strided_sliceStridedSlice-token_and_position_embedding_1/Shape:output:0;token_and_position_embedding_1/strided_slice/stack:output:0=token_and_position_embedding_1/strided_slice/stack_1:output:0=token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : l
*token_and_position_embedding_1/range/limitConst*
_output_shapes
: *
dtype0*
value	B :	l
*token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
$token_and_position_embedding_1/rangeRange3token_and_position_embedding_1/range/start:output:03token_and_position_embedding_1/range/limit:output:03token_and_position_embedding_1/range/delta:output:0*
_output_shapes
:	�
;token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherDtoken_and_position_embedding_1_embedding_3_embedding_lookup_12158911-token_and_position_embedding_1/range:output:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/12158911*
_output_shapes

:	*
dtype0�
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_3/embedding_lookup:output:0*
T0*W
_classM
KIloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/12158911*
_output_shapes

:	�
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	�
/token_and_position_embedding_1/embedding_2/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������	�
;token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherDtoken_and_position_embedding_1_embedding_2_embedding_lookup_121589173token_and_position_embedding_1/embedding_2/Cast:y:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/12158917*+
_output_shapes
:���������	*
dtype0�
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_2/embedding_lookup:output:0*
T0*W
_classM
KIloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/12158917*+
_output_shapes
:���������	�
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
"token_and_position_embedding_1/addAddV2Otoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	�
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: �
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
%batch_normalization_6/batchnorm/mul_1Muldense_11/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: �
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
@transformer_block_1/attention/query/einsum/Einsum/ReadVariableOpReadVariableOpItransformer_block_1_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
1transformer_block_1/attention/query/einsum/EinsumEinsum&token_and_position_embedding_1/add:z:0Htransformer_block_1/attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
6transformer_block_1/attention/query/add/ReadVariableOpReadVariableOp?transformer_block_1_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
'transformer_block_1/attention/query/addAddV2:transformer_block_1/attention/query/einsum/Einsum:output:0>transformer_block_1/attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
>transformer_block_1/attention/key/einsum/Einsum/ReadVariableOpReadVariableOpGtransformer_block_1_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
/transformer_block_1/attention/key/einsum/EinsumEinsum&token_and_position_embedding_1/add:z:0Ftransformer_block_1/attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
4transformer_block_1/attention/key/add/ReadVariableOpReadVariableOp=transformer_block_1_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
%transformer_block_1/attention/key/addAddV28transformer_block_1/attention/key/einsum/Einsum:output:0<transformer_block_1/attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
@transformer_block_1/attention/value/einsum/Einsum/ReadVariableOpReadVariableOpItransformer_block_1_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
1transformer_block_1/attention/value/einsum/EinsumEinsum&token_and_position_embedding_1/add:z:0Htransformer_block_1/attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
6transformer_block_1/attention/value/add/ReadVariableOpReadVariableOp?transformer_block_1_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
'transformer_block_1/attention/value/addAddV2:transformer_block_1/attention/value/einsum/Einsum:output:0>transformer_block_1/attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	h
#transformer_block_1/attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
!transformer_block_1/attention/MulMul+transformer_block_1/attention/query/add:z:0,transformer_block_1/attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
+transformer_block_1/attention/einsum/EinsumEinsum)transformer_block_1/attention/key/add:z:0%transformer_block_1/attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
-transformer_block_1/attention/softmax/SoftmaxSoftmax4transformer_block_1/attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
.transformer_block_1/attention/dropout/IdentityIdentity7transformer_block_1/attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������		�
-transformer_block_1/attention/einsum_1/EinsumEinsum7transformer_block_1/attention/dropout/Identity:output:0+transformer_block_1/attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
Ktransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_1_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
<transformer_block_1/attention/attention_output/einsum/EinsumEinsum6transformer_block_1/attention/einsum_1/Einsum:output:0Stransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
Atransformer_block_1/attention/attention_output/add/ReadVariableOpReadVariableOpJtransformer_block_1_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
2transformer_block_1/attention/attention_output/addAddV2Etransformer_block_1/attention/attention_output/einsum/Einsum:output:0Itransformer_block_1/attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&transformer_block_1/dropout_6/IdentityIdentity6transformer_block_1/attention/attention_output/add:z:0*
T0*+
_output_shapes
:���������	�
transformer_block_1/addAddV2&token_and_position_embedding_1/add:z:0/transformer_block_1/dropout_6/Identity:output:0*
T0*+
_output_shapes
:���������	�
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block_1/layer_normalization_2/moments/meanMeantransformer_block_1/add:z:0Qtransformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
>transformer_block_1/layer_normalization_2/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add:z:0Gtransformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block_1/layer_normalization_2/moments/varianceMeanGtransformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(~
9transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block_1/layer_normalization_2/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_2/moments/variance:output:0Btransformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
7transformer_block_1/layer_normalization_2/batchnorm/mulMul=transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_2/batchnorm/mul_1Multransformer_block_1/add:z:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_2/moments/mean:output:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
7transformer_block_1/layer_normalization_2/batchnorm/subSubJtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
Atransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
7transformer_block_1/sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_block_1/sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8transformer_block_1/sequential_1/dense_7/Tensordot/ShapeShape=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
@transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_7/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_7/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Btransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_7/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_7/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8transformer_block_1/sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block_1/sequential_1/dense_7/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:transformer_block_1/sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9transformer_block_1/sequential_1/dense_7/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>transformer_block_1/sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block_1/sequential_1/dense_7/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_7/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_7/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8transformer_block_1/sequential_1/dense_7/Tensordot/stackPack@transformer_block_1/sequential_1/dense_7/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<transformer_block_1/sequential_1/dense_7/Tensordot/transpose	Transpose=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Btransformer_block_1/sequential_1/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
:transformer_block_1/sequential_1/dense_7/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_7/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9transformer_block_1/sequential_1/dense_7/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_7/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
:transformer_block_1/sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
@transformer_block_1/sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block_1/sequential_1/dense_7/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_7/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_7/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2transformer_block_1/sequential_1/dense_7/TensordotReshapeCtransformer_block_1/sequential_1/dense_7/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
?transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
0transformer_block_1/sequential_1/dense_7/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_7/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
-transformer_block_1/sequential_1/dense_7/ReluRelu9transformer_block_1/sequential_1/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
Atransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
7transformer_block_1/sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_block_1/sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8transformer_block_1/sequential_1/dense_8/Tensordot/ShapeShape;transformer_block_1/sequential_1/dense_7/Relu:activations:0*
T0*
_output_shapes
:�
@transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_8/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_8/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Btransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_8/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_8/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8transformer_block_1/sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block_1/sequential_1/dense_8/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:transformer_block_1/sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9transformer_block_1/sequential_1/dense_8/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>transformer_block_1/sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block_1/sequential_1/dense_8/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_8/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_8/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8transformer_block_1/sequential_1/dense_8/Tensordot/stackPack@transformer_block_1/sequential_1/dense_8/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<transformer_block_1/sequential_1/dense_8/Tensordot/transpose	Transpose;transformer_block_1/sequential_1/dense_7/Relu:activations:0Btransformer_block_1/sequential_1/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
:transformer_block_1/sequential_1/dense_8/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_8/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9transformer_block_1/sequential_1/dense_8/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_8/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
:transformer_block_1/sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
@transformer_block_1/sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block_1/sequential_1/dense_8/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_8/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_8/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2transformer_block_1/sequential_1/dense_8/TensordotReshapeCtransformer_block_1/sequential_1/dense_8/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
?transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
0transformer_block_1/sequential_1/dense_8/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_8/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&transformer_block_1/dropout_7/IdentityIdentity9transformer_block_1/sequential_1/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
transformer_block_1/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0/transformer_block_1/dropout_7/Identity:output:0*
T0*+
_output_shapes
:���������	�
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block_1/layer_normalization_3/moments/meanMeantransformer_block_1/add_1:z:0Qtransformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
>transformer_block_1/layer_normalization_3/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add_1:z:0Gtransformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block_1/layer_normalization_3/moments/varianceMeanGtransformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(~
9transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block_1/layer_normalization_3/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_3/moments/variance:output:0Btransformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
7transformer_block_1/layer_normalization_3/batchnorm/mulMul=transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_3/batchnorm/mul_1Multransformer_block_1/add_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_3/moments/mean:output:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
7transformer_block_1/layer_normalization_3/batchnorm/subSubJtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
9transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	|
dropout_10/IdentityIdentity)batch_normalization_6/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� s
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d_1/MeanMean=transformer_block_1/layer_normalization_3/batchnorm/add_1:z:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_12/MatMulMatMuldropout_10/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_9/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: �
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
%batch_normalization_7/batchnorm/mul_1Muldense_12/Relu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: �
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: �
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: �
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� {
dropout_8/IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� |
dropout_11/IdentityIdentity)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� [
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2dropout_8/Identity:output:0dropout_11/Identity:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_13/MatMulMatMulconcatenate_1/concat:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp/^batch_normalization_6/batchnorm/ReadVariableOp1^batch_normalization_6/batchnorm/ReadVariableOp_11^batch_normalization_6/batchnorm/ReadVariableOp_23^batch_normalization_6/batchnorm/mul/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp1^batch_normalization_7/batchnorm/ReadVariableOp_11^batch_normalization_7/batchnorm/ReadVariableOp_23^batch_normalization_7/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp<^token_and_position_embedding_1/embedding_2/embedding_lookup<^token_and_position_embedding_1/embedding_3/embedding_lookupB^transformer_block_1/attention/attention_output/add/ReadVariableOpL^transformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp5^transformer_block_1/attention/key/add/ReadVariableOp?^transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp7^transformer_block_1/attention/query/add/ReadVariableOpA^transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp7^transformer_block_1/attention/value/add/ReadVariableOpA^transformer_block_1/attention/value/einsum/Einsum/ReadVariableOpC^transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpC^transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp@^transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp@^transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2d
0batch_normalization_6/batchnorm/ReadVariableOp_10batch_normalization_6/batchnorm/ReadVariableOp_12d
0batch_normalization_6/batchnorm/ReadVariableOp_20batch_normalization_6/batchnorm/ReadVariableOp_22h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2d
0batch_normalization_7/batchnorm/ReadVariableOp_10batch_normalization_7/batchnorm/ReadVariableOp_12d
0batch_normalization_7/batchnorm/ReadVariableOp_20batch_normalization_7/batchnorm/ReadVariableOp_22h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2z
;token_and_position_embedding_1/embedding_2/embedding_lookup;token_and_position_embedding_1/embedding_2/embedding_lookup2z
;token_and_position_embedding_1/embedding_3/embedding_lookup;token_and_position_embedding_1/embedding_3/embedding_lookup2�
Atransformer_block_1/attention/attention_output/add/ReadVariableOpAtransformer_block_1/attention/attention_output/add/ReadVariableOp2�
Ktransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOpKtransformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp2l
4transformer_block_1/attention/key/add/ReadVariableOp4transformer_block_1/attention/key/add/ReadVariableOp2�
>transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp>transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp2p
6transformer_block_1/attention/query/add/ReadVariableOp6transformer_block_1/attention/query/add/ReadVariableOp2�
@transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp@transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp2p
6transformer_block_1/attention/value/add/ReadVariableOp6transformer_block_1/attention/value/add/ReadVariableOp2�
@transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp@transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp2�
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2�
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2�
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2�
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2�
?transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp2�
Atransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp2�
?transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp2�
Atransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs/1
�
e
G__inference_dropout_8_layer_call_and_return_conditional_losses_12157560

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_6_layer_call_fn_12160120

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12157047o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12160378

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12157047

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
f
-__inference_dropout_11_layer_call_fn_12160449

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12157724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
t
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12157068

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12159733
x7
%embedding_3_embedding_lookup_12159720:	7
%embedding_2_embedding_lookup_12159726:
identity��embedding_2/embedding_lookup�embedding_3/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/limitConst*
_output_shapes
: *
dtype0*
value	B :	M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :l
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:	�
embedding_3/embedding_lookupResourceGather%embedding_3_embedding_lookup_12159720range:output:0*
Tindices0*8
_class.
,*loc:@embedding_3/embedding_lookup/12159720*
_output_shapes

:	*
dtype0�
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_3/embedding_lookup/12159720*
_output_shapes

:	�
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	\
embedding_2/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:���������	�
embedding_2/embedding_lookupResourceGather%embedding_2_embedding_lookup_12159726embedding_2/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_2/embedding_lookup/12159726*+
_output_shapes
:���������	*
dtype0�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_2/embedding_lookup/12159726*+
_output_shapes
:���������	�
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
addAddV20embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp^embedding_2/embedding_lookup^embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:J F
'
_output_shapes
:���������	

_user_specified_namex
�
�
8__inference_batch_normalization_4_layer_call_fn_12160278

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12157142o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�>
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_12160582

inputs;
)dense_7_tensordot_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:;
)dense_8_tensordot_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:
identity��dense_7/BiasAdd/ReadVariableOp� dense_7/Tensordot/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp� dense_8/Tensordot/ReadVariableOp�
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_7/Tensordot/transpose	Transposeinputs!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������c
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	d
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       a
dense_8/Tensordot/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_8/Tensordot/transpose	Transposedense_7/Relu:activations:0!dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	k
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12157177

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�f
�
E__inference_model_1_layer_call_and_return_conditional_losses_12158673
input_3
input_4#
dense_10_12158567:
 
dense_10_12158569: ,
batch_normalization_5_12158572: ,
batch_normalization_5_12158574: ,
batch_normalization_5_12158576: ,
batch_normalization_5_12158578: #
dense_11_12158582:  
dense_11_12158584: 9
'token_and_position_embedding_1_12158587:	9
'token_and_position_embedding_1_12158589:,
batch_normalization_6_12158592: ,
batch_normalization_6_12158594: ,
batch_normalization_6_12158596: ,
batch_normalization_6_12158598: 2
transformer_block_1_12158601:.
transformer_block_1_12158603:2
transformer_block_1_12158605:.
transformer_block_1_12158607:2
transformer_block_1_12158609:.
transformer_block_1_12158611:2
transformer_block_1_12158613:*
transformer_block_1_12158615:*
transformer_block_1_12158617:*
transformer_block_1_12158619:.
transformer_block_1_12158621:*
transformer_block_1_12158623:.
transformer_block_1_12158625:*
transformer_block_1_12158627:*
transformer_block_1_12158629:*
transformer_block_1_12158631:#
dense_12_12158636:  
dense_12_12158638: "
dense_9_12158641: 
dense_9_12158643: ,
batch_normalization_7_12158646: ,
batch_normalization_7_12158648: ,
batch_normalization_7_12158650: ,
batch_normalization_7_12158652: ,
batch_normalization_4_12158655: ,
batch_normalization_4_12158657: ,
batch_normalization_4_12158659: ,
batch_normalization_4_12158661: #
dense_13_12158667:@
dense_13_12158669:
identity��-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�"dropout_10/StatefulPartitionedCall�"dropout_11/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�6token_and_position_embedding_1/StatefulPartitionedCall�+transformer_block_1/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_10_12158567dense_10_12158569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_12157255�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_12158572batch_normalization_5_12158574batch_normalization_5_12158576batch_normalization_5_12158578*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12156772�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12158048�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_11_12158582dense_11_12158584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_12157288�
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_3'token_and_position_embedding_1_12158587'token_and_position_embedding_1_12158589*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *e
f`R^
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12157319�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_6_12158592batch_normalization_6_12158594batch_normalization_6_12158596batch_normalization_6_12158598*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12157047�
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0transformer_block_1_12158601transformer_block_1_12158603transformer_block_1_12158605transformer_block_1_12158607transformer_block_1_12158609transformer_block_1_12158611transformer_block_1_12158613transformer_block_1_12158615transformer_block_1_12158617transformer_block_1_12158619transformer_block_1_12158621transformer_block_1_12158623transformer_block_1_12158625transformer_block_1_12158627transformer_block_1_12158629transformer_block_1_12158631*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12157973�
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12157790�
*global_average_pooling1d_1/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12157068�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_12_12158636dense_12_12158638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_12157514�
dense_9/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_9_12158641dense_9_12158643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_12157531�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_7_12158646batch_normalization_7_12158648batch_normalization_7_12158650batch_normalization_7_12158652*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12157224�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_12158655batch_normalization_4_12158657batch_normalization_4_12158659batch_normalization_4_12158661*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12157142�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_12157747�
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12157724�
concatenate_1/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0+dropout_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12157576�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_13_12158667dense_13_12158669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_12157589x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������

!
_user_specified_name	input_4
�	
f
G__inference_dropout_8_layer_call_and_return_conditional_losses_12157747

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�

*__inference_model_1_layer_call_fn_12157687
input_3
input_4
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:  

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41:@

unknown_42:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_12157596o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������

!
_user_specified_name	input_4
�
�
*__inference_dense_7_layer_call_fn_12160648

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_12156821s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_4_layer_call_fn_12160265

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12157095o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
f
H__inference_dropout_11_layer_call_and_return_conditional_losses_12160454

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
g
H__inference_dropout_11_layer_call_and_return_conditional_losses_12160466

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_12_layer_call_and_return_conditional_losses_12160252

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
f
-__inference_dropout_10_layer_call_fn_12160195

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12157790o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
H
,__inference_dropout_9_layer_call_fn_12159677

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12157275`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
E__inference_dense_7_layer_call_and_return_conditional_losses_12156821

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������	e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������	z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156864

inputs"
dense_7_12156822:
dense_7_12156824:"
dense_8_12156858:
dense_8_12156860:
identity��dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_12156822dense_7_12156824*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_12156821�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_12156858dense_8_12156860*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_12156857{
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�

*__inference_model_1_layer_call_fn_12158867
inputs_0
inputs_1
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:  

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41:@

unknown_42:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#&'*+,-*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_12158268o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs/1
�
H
,__inference_dropout_8_layer_call_fn_12160417

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_12157560`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_13_layer_call_and_return_conditional_losses_12157589

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
e
G__inference_dropout_9_layer_call_and_return_conditional_losses_12159687

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_1_layer_call_fn_12160512

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156864s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12160298

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
f
G__inference_dropout_9_layer_call_and_return_conditional_losses_12158048

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12157319
x7
%embedding_3_embedding_lookup_12157306:	7
%embedding_2_embedding_lookup_12157312:
identity��embedding_2/embedding_lookup�embedding_3/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/limitConst*
_output_shapes
: *
dtype0*
value	B :	M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :l
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:	�
embedding_3/embedding_lookupResourceGather%embedding_3_embedding_lookup_12157306range:output:0*
Tindices0*8
_class.
,*loc:@embedding_3/embedding_lookup/12157306*
_output_shapes

:	*
dtype0�
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_3/embedding_lookup/12157306*
_output_shapes

:	�
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	\
embedding_2/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:���������	�
embedding_2/embedding_lookupResourceGather%embedding_2_embedding_lookup_12157312embedding_2/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_2/embedding_lookup/12157312*+
_output_shapes
:���������	*
dtype0�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_2/embedding_lookup/12157312*+
_output_shapes
:���������	�
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
addAddV20embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp^embedding_2/embedding_lookup^embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:J F
'
_output_shapes
:���������	

_user_specified_namex
�
�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12157000

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
u
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12157576

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:��������� :��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
e
G__inference_dropout_8_layer_call_and_return_conditional_losses_12160427

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�`
�
E__inference_model_1_layer_call_and_return_conditional_losses_12158563
input_3
input_4#
dense_10_12158457:
 
dense_10_12158459: ,
batch_normalization_5_12158462: ,
batch_normalization_5_12158464: ,
batch_normalization_5_12158466: ,
batch_normalization_5_12158468: #
dense_11_12158472:  
dense_11_12158474: 9
'token_and_position_embedding_1_12158477:	9
'token_and_position_embedding_1_12158479:,
batch_normalization_6_12158482: ,
batch_normalization_6_12158484: ,
batch_normalization_6_12158486: ,
batch_normalization_6_12158488: 2
transformer_block_1_12158491:.
transformer_block_1_12158493:2
transformer_block_1_12158495:.
transformer_block_1_12158497:2
transformer_block_1_12158499:.
transformer_block_1_12158501:2
transformer_block_1_12158503:*
transformer_block_1_12158505:*
transformer_block_1_12158507:*
transformer_block_1_12158509:.
transformer_block_1_12158511:*
transformer_block_1_12158513:.
transformer_block_1_12158515:*
transformer_block_1_12158517:*
transformer_block_1_12158519:*
transformer_block_1_12158521:#
dense_12_12158526:  
dense_12_12158528: "
dense_9_12158531: 
dense_9_12158533: ,
batch_normalization_7_12158536: ,
batch_normalization_7_12158538: ,
batch_normalization_7_12158540: ,
batch_normalization_7_12158542: ,
batch_normalization_4_12158545: ,
batch_normalization_4_12158547: ,
batch_normalization_4_12158549: ,
batch_normalization_4_12158551: #
dense_13_12158557:@
dense_13_12158559:
identity��-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�6token_and_position_embedding_1/StatefulPartitionedCall�+transformer_block_1/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_10_12158457dense_10_12158459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_12157255�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_12158462batch_normalization_5_12158464batch_normalization_5_12158466batch_normalization_5_12158468*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12156725�
dropout_9/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12157275�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_11_12158472dense_11_12158474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_12157288�
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_3'token_and_position_embedding_1_12158477'token_and_position_embedding_1_12158479*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *e
f`R^
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12157319�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_6_12158482batch_normalization_6_12158484batch_normalization_6_12158486batch_normalization_6_12158488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12157000�
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0transformer_block_1_12158491transformer_block_1_12158493transformer_block_1_12158495transformer_block_1_12158497transformer_block_1_12158499transformer_block_1_12158501transformer_block_1_12158503transformer_block_1_12158505transformer_block_1_12158507transformer_block_1_12158509transformer_block_1_12158511transformer_block_1_12158513transformer_block_1_12158515transformer_block_1_12158517transformer_block_1_12158519transformer_block_1_12158521*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12157461�
dropout_10/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12157500�
*global_average_pooling1d_1/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12157068�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_12_12158526dense_12_12158528*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_12157514�
dense_9/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_9_12158531dense_9_12158533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_12157531�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_7_12158536batch_normalization_7_12158538batch_normalization_7_12158540batch_normalization_7_12158542*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12157177�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_12158545batch_normalization_4_12158547batch_normalization_4_12158549batch_normalization_4_12158551*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12157095�
dropout_8/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_12157560�
dropout_11/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12157567�
concatenate_1/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0#dropout_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12157576�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_13_12158557dense_13_12158559*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_12157589x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������

!
_user_specified_name	input_4
�
�
E__inference_dense_7_layer_call_and_return_conditional_losses_12160679

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������	e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������	z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
g
H__inference_dropout_10_layer_call_and_return_conditional_losses_12157790

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
g
H__inference_dropout_11_layer_call_and_return_conditional_losses_12157724

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12156772

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
f
H__inference_dropout_10_layer_call_and_return_conditional_losses_12157500

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_12_layer_call_and_return_conditional_losses_12157514

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�

&__inference_signature_wrapper_12159572
input_3
input_4
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:  

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41:@

unknown_42:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_12156701o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������

!
_user_specified_name	input_4
�
�
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12159954

inputsK
5attention_query_einsum_einsum_readvariableop_resource:=
+attention_query_add_readvariableop_resource:I
3attention_key_einsum_einsum_readvariableop_resource:;
)attention_key_add_readvariableop_resource:K
5attention_value_einsum_einsum_readvariableop_resource:=
+attention_value_add_readvariableop_resource:V
@attention_attention_output_einsum_einsum_readvariableop_resource:D
6attention_attention_output_add_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:H
6sequential_1_dense_7_tensordot_readvariableop_resource:B
4sequential_1_dense_7_biasadd_readvariableop_resource:H
6sequential_1_dense_8_tensordot_readvariableop_resource:B
4sequential_1_dense_8_biasadd_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:
identity��-attention/attention_output/add/ReadVariableOp�7attention/attention_output/einsum/Einsum/ReadVariableOp� attention/key/add/ReadVariableOp�*attention/key/einsum/Einsum/ReadVariableOp�"attention/query/add/ReadVariableOp�,attention/query/einsum/Einsum/ReadVariableOp�"attention/value/add/ReadVariableOp�,attention/value/einsum/Einsum/ReadVariableOp�.layer_normalization_2/batchnorm/ReadVariableOp�2layer_normalization_2/batchnorm/mul/ReadVariableOp�.layer_normalization_3/batchnorm/ReadVariableOp�2layer_normalization_3/batchnorm/mul/ReadVariableOp�+sequential_1/dense_7/BiasAdd/ReadVariableOp�-sequential_1/dense_7/Tensordot/ReadVariableOp�+sequential_1/dense_8/BiasAdd/ReadVariableOp�-sequential_1/dense_8/Tensordot/ReadVariableOp�
,attention/query/einsum/Einsum/ReadVariableOpReadVariableOp5attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/query/einsum/EinsumEinsuminputs4attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
"attention/query/add/ReadVariableOpReadVariableOp+attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/query/addAddV2&attention/query/einsum/Einsum:output:0*attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
*attention/key/einsum/Einsum/ReadVariableOpReadVariableOp3attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/key/einsum/EinsumEinsuminputs2attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
 attention/key/add/ReadVariableOpReadVariableOp)attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/key/addAddV2$attention/key/einsum/Einsum:output:0(attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
,attention/value/einsum/Einsum/ReadVariableOpReadVariableOp5attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/value/einsum/EinsumEinsuminputs4attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
"attention/value/add/ReadVariableOpReadVariableOp+attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/value/addAddV2&attention/value/einsum/Einsum:output:0*attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	T
attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
attention/MulMulattention/query/add:z:0attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
attention/einsum/EinsumEinsumattention/key/add:z:0attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
attention/softmax/SoftmaxSoftmax attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
attention/dropout/IdentityIdentity#attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������		�
attention/einsum_1/EinsumEinsum#attention/dropout/Identity:output:0attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
7attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp@attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
(attention/attention_output/einsum/EinsumEinsum"attention/einsum_1/Einsum:output:0?attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
-attention/attention_output/add/ReadVariableOpReadVariableOp6attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention/attention_output/addAddV21attention/attention_output/einsum/Einsum:output:05attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	x
dropout_6/IdentityIdentity"attention/attention_output/add:z:0*
T0*+
_output_shapes
:���������	g
addAddV2inputsdropout_6/Identity:output:0*
T0*+
_output_shapes
:���������	~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_1/dense_7/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_7/Tensordot/GatherV2GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/free:output:05sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/dense_7/Tensordot/GatherV2_1GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/axes:output:07sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	~
sequential_1/dense_7/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
$sequential_1/dense_8/Tensordot/ShapeShape'sequential_1/dense_7/Relu:activations:0*
T0*
_output_shapes
:n
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/dense_8/Tensordot/transpose	Transpose'sequential_1/dense_7/Relu:activations:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	{
dropout_7/IdentityIdentity%sequential_1/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_7/Identity:output:0*
T0*+
_output_shapes
:���������	~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	|
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������	: : : : : : : : : : : : : : : : 2^
-attention/attention_output/add/ReadVariableOp-attention/attention_output/add/ReadVariableOp2r
7attention/attention_output/einsum/Einsum/ReadVariableOp7attention/attention_output/einsum/Einsum/ReadVariableOp2D
 attention/key/add/ReadVariableOp attention/key/add/ReadVariableOp2X
*attention/key/einsum/Einsum/ReadVariableOp*attention/key/einsum/Einsum/ReadVariableOp2H
"attention/query/add/ReadVariableOp"attention/query/add/ReadVariableOp2\
,attention/query/einsum/Einsum/ReadVariableOp,attention/query/einsum/Einsum/ReadVariableOp2H
"attention/value/add/ReadVariableOp"attention/value/add/ReadVariableOp2\
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2^
-sequential_1/dense_7/Tensordot/ReadVariableOp-sequential_1/dense_7/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12160174

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
e
,__inference_dropout_8_layer_call_fn_12160422

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_12157747o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_1_layer_call_fn_12156875
dense_7_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156864s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������	
'
_user_specified_namedense_7_input
��
�Y
$__inference__traced_restore_12161490
file_prefix2
 assignvariableop_dense_10_kernel:
 .
 assignvariableop_1_dense_10_bias: <
.assignvariableop_2_batch_normalization_5_gamma: ;
-assignvariableop_3_batch_normalization_5_beta: B
4assignvariableop_4_batch_normalization_5_moving_mean: F
8assignvariableop_5_batch_normalization_5_moving_variance: 4
"assignvariableop_6_dense_11_kernel:  .
 assignvariableop_7_dense_11_bias: <
.assignvariableop_8_batch_normalization_6_gamma: ;
-assignvariableop_9_batch_normalization_6_beta: C
5assignvariableop_10_batch_normalization_6_moving_mean: G
9assignvariableop_11_batch_normalization_6_moving_variance: 4
"assignvariableop_12_dense_9_kernel: .
 assignvariableop_13_dense_9_bias: 5
#assignvariableop_14_dense_12_kernel:  /
!assignvariableop_15_dense_12_bias: =
/assignvariableop_16_batch_normalization_4_gamma: <
.assignvariableop_17_batch_normalization_4_beta: C
5assignvariableop_18_batch_normalization_4_moving_mean: G
9assignvariableop_19_batch_normalization_4_moving_variance: =
/assignvariableop_20_batch_normalization_7_gamma: <
.assignvariableop_21_batch_normalization_7_beta: C
5assignvariableop_22_batch_normalization_7_moving_mean: G
9assignvariableop_23_batch_normalization_7_moving_variance: 5
#assignvariableop_24_dense_13_kernel:@/
!assignvariableop_25_dense_13_bias:$
assignvariableop_26_beta_1: $
assignvariableop_27_beta_2: #
assignvariableop_28_decay: +
!assignvariableop_29_learning_rate: '
assignvariableop_30_adam_iter:	 [
Iassignvariableop_31_token_and_position_embedding_1_embedding_2_embeddings:[
Iassignvariableop_32_token_and_position_embedding_1_embedding_3_embeddings:	T
>assignvariableop_33_transformer_block_1_attention_query_kernel:N
<assignvariableop_34_transformer_block_1_attention_query_bias:R
<assignvariableop_35_transformer_block_1_attention_key_kernel:L
:assignvariableop_36_transformer_block_1_attention_key_bias:T
>assignvariableop_37_transformer_block_1_attention_value_kernel:N
<assignvariableop_38_transformer_block_1_attention_value_bias:_
Iassignvariableop_39_transformer_block_1_attention_attention_output_kernel:U
Gassignvariableop_40_transformer_block_1_attention_attention_output_bias:4
"assignvariableop_41_dense_7_kernel:.
 assignvariableop_42_dense_7_bias:4
"assignvariableop_43_dense_8_kernel:.
 assignvariableop_44_dense_8_bias:Q
Cassignvariableop_45_transformer_block_1_layer_normalization_2_gamma:P
Bassignvariableop_46_transformer_block_1_layer_normalization_2_beta:Q
Cassignvariableop_47_transformer_block_1_layer_normalization_3_gamma:P
Bassignvariableop_48_transformer_block_1_layer_normalization_3_beta:#
assignvariableop_49_total: #
assignvariableop_50_count: <
*assignvariableop_51_adam_dense_10_kernel_m:
 6
(assignvariableop_52_adam_dense_10_bias_m: D
6assignvariableop_53_adam_batch_normalization_5_gamma_m: C
5assignvariableop_54_adam_batch_normalization_5_beta_m: <
*assignvariableop_55_adam_dense_11_kernel_m:  6
(assignvariableop_56_adam_dense_11_bias_m: D
6assignvariableop_57_adam_batch_normalization_6_gamma_m: C
5assignvariableop_58_adam_batch_normalization_6_beta_m: ;
)assignvariableop_59_adam_dense_9_kernel_m: 5
'assignvariableop_60_adam_dense_9_bias_m: <
*assignvariableop_61_adam_dense_12_kernel_m:  6
(assignvariableop_62_adam_dense_12_bias_m: D
6assignvariableop_63_adam_batch_normalization_4_gamma_m: C
5assignvariableop_64_adam_batch_normalization_4_beta_m: D
6assignvariableop_65_adam_batch_normalization_7_gamma_m: C
5assignvariableop_66_adam_batch_normalization_7_beta_m: <
*assignvariableop_67_adam_dense_13_kernel_m:@6
(assignvariableop_68_adam_dense_13_bias_m:b
Passignvariableop_69_adam_token_and_position_embedding_1_embedding_2_embeddings_m:b
Passignvariableop_70_adam_token_and_position_embedding_1_embedding_3_embeddings_m:	[
Eassignvariableop_71_adam_transformer_block_1_attention_query_kernel_m:U
Cassignvariableop_72_adam_transformer_block_1_attention_query_bias_m:Y
Cassignvariableop_73_adam_transformer_block_1_attention_key_kernel_m:S
Aassignvariableop_74_adam_transformer_block_1_attention_key_bias_m:[
Eassignvariableop_75_adam_transformer_block_1_attention_value_kernel_m:U
Cassignvariableop_76_adam_transformer_block_1_attention_value_bias_m:f
Passignvariableop_77_adam_transformer_block_1_attention_attention_output_kernel_m:\
Nassignvariableop_78_adam_transformer_block_1_attention_attention_output_bias_m:;
)assignvariableop_79_adam_dense_7_kernel_m:5
'assignvariableop_80_adam_dense_7_bias_m:;
)assignvariableop_81_adam_dense_8_kernel_m:5
'assignvariableop_82_adam_dense_8_bias_m:X
Jassignvariableop_83_adam_transformer_block_1_layer_normalization_2_gamma_m:W
Iassignvariableop_84_adam_transformer_block_1_layer_normalization_2_beta_m:X
Jassignvariableop_85_adam_transformer_block_1_layer_normalization_3_gamma_m:W
Iassignvariableop_86_adam_transformer_block_1_layer_normalization_3_beta_m:<
*assignvariableop_87_adam_dense_10_kernel_v:
 6
(assignvariableop_88_adam_dense_10_bias_v: D
6assignvariableop_89_adam_batch_normalization_5_gamma_v: C
5assignvariableop_90_adam_batch_normalization_5_beta_v: <
*assignvariableop_91_adam_dense_11_kernel_v:  6
(assignvariableop_92_adam_dense_11_bias_v: D
6assignvariableop_93_adam_batch_normalization_6_gamma_v: C
5assignvariableop_94_adam_batch_normalization_6_beta_v: ;
)assignvariableop_95_adam_dense_9_kernel_v: 5
'assignvariableop_96_adam_dense_9_bias_v: <
*assignvariableop_97_adam_dense_12_kernel_v:  6
(assignvariableop_98_adam_dense_12_bias_v: D
6assignvariableop_99_adam_batch_normalization_4_gamma_v: D
6assignvariableop_100_adam_batch_normalization_4_beta_v: E
7assignvariableop_101_adam_batch_normalization_7_gamma_v: D
6assignvariableop_102_adam_batch_normalization_7_beta_v: =
+assignvariableop_103_adam_dense_13_kernel_v:@7
)assignvariableop_104_adam_dense_13_bias_v:c
Qassignvariableop_105_adam_token_and_position_embedding_1_embedding_2_embeddings_v:c
Qassignvariableop_106_adam_token_and_position_embedding_1_embedding_3_embeddings_v:	\
Fassignvariableop_107_adam_transformer_block_1_attention_query_kernel_v:V
Dassignvariableop_108_adam_transformer_block_1_attention_query_bias_v:Z
Dassignvariableop_109_adam_transformer_block_1_attention_key_kernel_v:T
Bassignvariableop_110_adam_transformer_block_1_attention_key_bias_v:\
Fassignvariableop_111_adam_transformer_block_1_attention_value_kernel_v:V
Dassignvariableop_112_adam_transformer_block_1_attention_value_bias_v:g
Qassignvariableop_113_adam_transformer_block_1_attention_attention_output_kernel_v:]
Oassignvariableop_114_adam_transformer_block_1_attention_attention_output_bias_v:<
*assignvariableop_115_adam_dense_7_kernel_v:6
(assignvariableop_116_adam_dense_7_bias_v:<
*assignvariableop_117_adam_dense_8_kernel_v:6
(assignvariableop_118_adam_dense_8_bias_v:Y
Kassignvariableop_119_adam_transformer_block_1_layer_normalization_2_gamma_v:X
Jassignvariableop_120_adam_transformer_block_1_layer_normalization_2_beta_v:Y
Kassignvariableop_121_adam_transformer_block_1_layer_normalization_3_gamma_v:X
Jassignvariableop_122_adam_transformer_block_1_layer_normalization_3_beta_v:
identity_124��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:|*
dtype0*�?
value�?B�?|B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:|*
dtype0*�
value�B�|B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
~2|	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_5_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_5_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_5_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_5_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_6_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_6_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_6_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_6_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_9_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_9_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_12_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_12_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_4_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_4_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_4_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_4_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_7_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_7_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_7_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_7_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_13_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_13_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_beta_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_beta_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_decayIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_learning_rateIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpIassignvariableop_31_token_and_position_embedding_1_embedding_2_embeddingsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpIassignvariableop_32_token_and_position_embedding_1_embedding_3_embeddingsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp>assignvariableop_33_transformer_block_1_attention_query_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp<assignvariableop_34_transformer_block_1_attention_query_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp<assignvariableop_35_transformer_block_1_attention_key_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp:assignvariableop_36_transformer_block_1_attention_key_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp>assignvariableop_37_transformer_block_1_attention_value_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp<assignvariableop_38_transformer_block_1_attention_value_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpIassignvariableop_39_transformer_block_1_attention_attention_output_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpGassignvariableop_40_transformer_block_1_attention_attention_output_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp"assignvariableop_41_dense_7_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp assignvariableop_42_dense_7_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp"assignvariableop_43_dense_8_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp assignvariableop_44_dense_8_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpCassignvariableop_45_transformer_block_1_layer_normalization_2_gammaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpBassignvariableop_46_transformer_block_1_layer_normalization_2_betaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpCassignvariableop_47_transformer_block_1_layer_normalization_3_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpBassignvariableop_48_transformer_block_1_layer_normalization_3_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_10_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_10_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_5_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_5_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_11_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_11_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_6_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_6_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_9_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_9_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_12_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_12_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_batch_normalization_4_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp5assignvariableop_64_adam_batch_normalization_4_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_7_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_7_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_13_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_13_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpPassignvariableop_69_adam_token_and_position_embedding_1_embedding_2_embeddings_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOpPassignvariableop_70_adam_token_and_position_embedding_1_embedding_3_embeddings_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOpEassignvariableop_71_adam_transformer_block_1_attention_query_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOpCassignvariableop_72_adam_transformer_block_1_attention_query_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOpCassignvariableop_73_adam_transformer_block_1_attention_key_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpAassignvariableop_74_adam_transformer_block_1_attention_key_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpEassignvariableop_75_adam_transformer_block_1_attention_value_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpCassignvariableop_76_adam_transformer_block_1_attention_value_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpPassignvariableop_77_adam_transformer_block_1_attention_attention_output_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpNassignvariableop_78_adam_transformer_block_1_attention_attention_output_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp)assignvariableop_79_adam_dense_7_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp'assignvariableop_80_adam_dense_7_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adam_dense_8_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp'assignvariableop_82_adam_dense_8_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOpJassignvariableop_83_adam_transformer_block_1_layer_normalization_2_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOpIassignvariableop_84_adam_transformer_block_1_layer_normalization_2_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOpJassignvariableop_85_adam_transformer_block_1_layer_normalization_3_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpIassignvariableop_86_adam_transformer_block_1_layer_normalization_3_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_10_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_10_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp6assignvariableop_89_adam_batch_normalization_5_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp5assignvariableop_90_adam_batch_normalization_5_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_11_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_11_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_batch_normalization_6_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_batch_normalization_6_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp)assignvariableop_95_adam_dense_9_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp'assignvariableop_96_adam_dense_9_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_dense_12_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_dense_12_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp6assignvariableop_99_adam_batch_normalization_4_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp6assignvariableop_100_adam_batch_normalization_4_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_batch_normalization_7_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp6assignvariableop_102_adam_batch_normalization_7_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_dense_13_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_dense_13_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOpQassignvariableop_105_adam_token_and_position_embedding_1_embedding_2_embeddings_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOpQassignvariableop_106_adam_token_and_position_embedding_1_embedding_3_embeddings_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOpFassignvariableop_107_adam_transformer_block_1_attention_query_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOpDassignvariableop_108_adam_transformer_block_1_attention_query_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOpDassignvariableop_109_adam_transformer_block_1_attention_key_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOpBassignvariableop_110_adam_transformer_block_1_attention_key_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOpFassignvariableop_111_adam_transformer_block_1_attention_value_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOpDassignvariableop_112_adam_transformer_block_1_attention_value_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOpQassignvariableop_113_adam_transformer_block_1_attention_attention_output_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOpOassignvariableop_114_adam_transformer_block_1_attention_attention_output_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp*assignvariableop_115_adam_dense_7_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp(assignvariableop_116_adam_dense_7_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp*assignvariableop_117_adam_dense_8_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp(assignvariableop_118_adam_dense_8_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOpKassignvariableop_119_adam_transformer_block_1_layer_normalization_2_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOpJassignvariableop_120_adam_transformer_block_1_layer_normalization_2_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOpKassignvariableop_121_adam_transformer_block_1_layer_normalization_3_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOpJassignvariableop_122_adam_transformer_block_1_layer_normalization_3_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_123Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_124IdentityIdentity_123:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_124Identity_124:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12157095

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12160094

inputsK
5attention_query_einsum_einsum_readvariableop_resource:=
+attention_query_add_readvariableop_resource:I
3attention_key_einsum_einsum_readvariableop_resource:;
)attention_key_add_readvariableop_resource:K
5attention_value_einsum_einsum_readvariableop_resource:=
+attention_value_add_readvariableop_resource:V
@attention_attention_output_einsum_einsum_readvariableop_resource:D
6attention_attention_output_add_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:H
6sequential_1_dense_7_tensordot_readvariableop_resource:B
4sequential_1_dense_7_biasadd_readvariableop_resource:H
6sequential_1_dense_8_tensordot_readvariableop_resource:B
4sequential_1_dense_8_biasadd_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:
identity��-attention/attention_output/add/ReadVariableOp�7attention/attention_output/einsum/Einsum/ReadVariableOp� attention/key/add/ReadVariableOp�*attention/key/einsum/Einsum/ReadVariableOp�"attention/query/add/ReadVariableOp�,attention/query/einsum/Einsum/ReadVariableOp�"attention/value/add/ReadVariableOp�,attention/value/einsum/Einsum/ReadVariableOp�.layer_normalization_2/batchnorm/ReadVariableOp�2layer_normalization_2/batchnorm/mul/ReadVariableOp�.layer_normalization_3/batchnorm/ReadVariableOp�2layer_normalization_3/batchnorm/mul/ReadVariableOp�+sequential_1/dense_7/BiasAdd/ReadVariableOp�-sequential_1/dense_7/Tensordot/ReadVariableOp�+sequential_1/dense_8/BiasAdd/ReadVariableOp�-sequential_1/dense_8/Tensordot/ReadVariableOp�
,attention/query/einsum/Einsum/ReadVariableOpReadVariableOp5attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/query/einsum/EinsumEinsuminputs4attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
"attention/query/add/ReadVariableOpReadVariableOp+attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/query/addAddV2&attention/query/einsum/Einsum:output:0*attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
*attention/key/einsum/Einsum/ReadVariableOpReadVariableOp3attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/key/einsum/EinsumEinsuminputs2attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
 attention/key/add/ReadVariableOpReadVariableOp)attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/key/addAddV2$attention/key/einsum/Einsum:output:0(attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
,attention/value/einsum/Einsum/ReadVariableOpReadVariableOp5attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/value/einsum/EinsumEinsuminputs4attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
"attention/value/add/ReadVariableOpReadVariableOp+attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/value/addAddV2&attention/value/einsum/Einsum:output:0*attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	T
attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
attention/MulMulattention/query/add:z:0attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
attention/einsum/EinsumEinsumattention/key/add:z:0attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
attention/softmax/SoftmaxSoftmax attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
attention/einsum_1/EinsumEinsum#attention/softmax/Softmax:softmax:0attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
7attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp@attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
(attention/attention_output/einsum/EinsumEinsum"attention/einsum_1/Einsum:output:0?attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
-attention/attention_output/add/ReadVariableOpReadVariableOp6attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention/attention_output/addAddV21attention/attention_output/einsum/Einsum:output:05attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_6/dropout/MulMul"attention/attention_output/add:z:0 dropout_6/dropout/Const:output:0*
T0*+
_output_shapes
:���������	i
dropout_6/dropout/ShapeShape"attention/attention_output/add:z:0*
T0*
_output_shapes
:�
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	g
addAddV2inputsdropout_6/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_1/dense_7/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_7/Tensordot/GatherV2GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/free:output:05sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/dense_7/Tensordot/GatherV2_1GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/axes:output:07sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	~
sequential_1/dense_7/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
$sequential_1/dense_8/Tensordot/ShapeShape'sequential_1/dense_7/Relu:activations:0*
T0*
_output_shapes
:n
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/dense_8/Tensordot/transpose	Transpose'sequential_1/dense_7/Relu:activations:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_7/dropout/MulMul%sequential_1/dense_8/BiasAdd:output:0 dropout_7/dropout/Const:output:0*
T0*+
_output_shapes
:���������	l
dropout_7/dropout/ShapeShape%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:�
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	�
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_7/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	|
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������	: : : : : : : : : : : : : : : : 2^
-attention/attention_output/add/ReadVariableOp-attention/attention_output/add/ReadVariableOp2r
7attention/attention_output/einsum/Einsum/ReadVariableOp7attention/attention_output/einsum/Einsum/ReadVariableOp2D
 attention/key/add/ReadVariableOp attention/key/add/ReadVariableOp2X
*attention/key/einsum/Einsum/ReadVariableOp*attention/key/einsum/Einsum/ReadVariableOp2H
"attention/query/add/ReadVariableOp"attention/query/add/ReadVariableOp2\
,attention/query/einsum/Einsum/ReadVariableOp,attention/query/einsum/Einsum/ReadVariableOp2H
"attention/value/add/ReadVariableOp"attention/value/add/ReadVariableOp2\
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2^
-sequential_1/dense_7/Tensordot/ReadVariableOp-sequential_1/dense_7/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
f
G__inference_dropout_8_layer_call_and_return_conditional_losses_12160439

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12160332

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_7_layer_call_fn_12160345

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12157177o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_5_layer_call_fn_12159618

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12156772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
E__inference_dense_8_layer_call_and_return_conditional_losses_12160718

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������	z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12157461

inputsK
5attention_query_einsum_einsum_readvariableop_resource:=
+attention_query_add_readvariableop_resource:I
3attention_key_einsum_einsum_readvariableop_resource:;
)attention_key_add_readvariableop_resource:K
5attention_value_einsum_einsum_readvariableop_resource:=
+attention_value_add_readvariableop_resource:V
@attention_attention_output_einsum_einsum_readvariableop_resource:D
6attention_attention_output_add_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:H
6sequential_1_dense_7_tensordot_readvariableop_resource:B
4sequential_1_dense_7_biasadd_readvariableop_resource:H
6sequential_1_dense_8_tensordot_readvariableop_resource:B
4sequential_1_dense_8_biasadd_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:
identity��-attention/attention_output/add/ReadVariableOp�7attention/attention_output/einsum/Einsum/ReadVariableOp� attention/key/add/ReadVariableOp�*attention/key/einsum/Einsum/ReadVariableOp�"attention/query/add/ReadVariableOp�,attention/query/einsum/Einsum/ReadVariableOp�"attention/value/add/ReadVariableOp�,attention/value/einsum/Einsum/ReadVariableOp�.layer_normalization_2/batchnorm/ReadVariableOp�2layer_normalization_2/batchnorm/mul/ReadVariableOp�.layer_normalization_3/batchnorm/ReadVariableOp�2layer_normalization_3/batchnorm/mul/ReadVariableOp�+sequential_1/dense_7/BiasAdd/ReadVariableOp�-sequential_1/dense_7/Tensordot/ReadVariableOp�+sequential_1/dense_8/BiasAdd/ReadVariableOp�-sequential_1/dense_8/Tensordot/ReadVariableOp�
,attention/query/einsum/Einsum/ReadVariableOpReadVariableOp5attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/query/einsum/EinsumEinsuminputs4attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
"attention/query/add/ReadVariableOpReadVariableOp+attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/query/addAddV2&attention/query/einsum/Einsum:output:0*attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
*attention/key/einsum/Einsum/ReadVariableOpReadVariableOp3attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/key/einsum/EinsumEinsuminputs2attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
 attention/key/add/ReadVariableOpReadVariableOp)attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/key/addAddV2$attention/key/einsum/Einsum:output:0(attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
,attention/value/einsum/Einsum/ReadVariableOpReadVariableOp5attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/value/einsum/EinsumEinsuminputs4attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
"attention/value/add/ReadVariableOpReadVariableOp+attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/value/addAddV2&attention/value/einsum/Einsum:output:0*attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	T
attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
attention/MulMulattention/query/add:z:0attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
attention/einsum/EinsumEinsumattention/key/add:z:0attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
attention/softmax/SoftmaxSoftmax attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
attention/dropout/IdentityIdentity#attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������		�
attention/einsum_1/EinsumEinsum#attention/dropout/Identity:output:0attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
7attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp@attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
(attention/attention_output/einsum/EinsumEinsum"attention/einsum_1/Einsum:output:0?attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
-attention/attention_output/add/ReadVariableOpReadVariableOp6attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention/attention_output/addAddV21attention/attention_output/einsum/Einsum:output:05attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	x
dropout_6/IdentityIdentity"attention/attention_output/add:z:0*
T0*+
_output_shapes
:���������	g
addAddV2inputsdropout_6/Identity:output:0*
T0*+
_output_shapes
:���������	~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_1/dense_7/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_7/Tensordot/GatherV2GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/free:output:05sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/dense_7/Tensordot/GatherV2_1GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/axes:output:07sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	~
sequential_1/dense_7/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
$sequential_1/dense_8/Tensordot/ShapeShape'sequential_1/dense_7/Relu:activations:0*
T0*
_output_shapes
:n
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/dense_8/Tensordot/transpose	Transpose'sequential_1/dense_7/Relu:activations:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	{
dropout_7/IdentityIdentity%sequential_1/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_7/Identity:output:0*
T0*+
_output_shapes
:���������	~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	|
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������	: : : : : : : : : : : : : : : : 2^
-attention/attention_output/add/ReadVariableOp-attention/attention_output/add/ReadVariableOp2r
7attention/attention_output/einsum/Einsum/ReadVariableOp7attention/attention_output/einsum/Einsum/ReadVariableOp2D
 attention/key/add/ReadVariableOp attention/key/add/ReadVariableOp2X
*attention/key/einsum/Einsum/ReadVariableOp*attention/key/einsum/Einsum/ReadVariableOp2H
"attention/query/add/ReadVariableOp"attention/query/add/ReadVariableOp2\
,attention/query/einsum/Einsum/ReadVariableOp,attention/query/einsum/Einsum/ReadVariableOp2H
"attention/value/add/ReadVariableOp"attention/value/add/ReadVariableOp2\
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2^
-sequential_1/dense_7/Tensordot/ReadVariableOp-sequential_1/dense_7/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
I
-__inference_dropout_10_layer_call_fn_12160190

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12157500`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156962
dense_7_input"
dense_7_12156951:
dense_7_12156953:"
dense_8_12156956:
dense_8_12156958:
identity��dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_12156951dense_7_12156953*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_12156821�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_12156956dense_8_12156958*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_12156857{
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Z V
+
_output_shapes
:���������	
'
_user_specified_namedense_7_input
�
�
+__inference_dense_11_layer_call_fn_12159742

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_12157288o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
I
-__inference_dropout_11_layer_call_fn_12160444

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12157567`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
f
G__inference_dropout_9_layer_call_and_return_conditional_losses_12159699

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_dense_8_layer_call_fn_12160688

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_12156857s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�

*__inference_model_1_layer_call_fn_12158773
inputs_0
inputs_1
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:  

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41:@

unknown_42:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_12157596o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs/1
�
�

*__inference_model_1_layer_call_fn_12158453
input_3
input_4
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:  

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41:@

unknown_42:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#&'*+,-*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_12158268o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������

!
_user_specified_name	input_4
�
�
6__inference_transformer_block_1_layer_call_fn_12159827

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12157973s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������	: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�>
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_12160639

inputs;
)dense_7_tensordot_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:;
)dense_8_tensordot_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:
identity��dense_7/BiasAdd/ReadVariableOp� dense_7/Tensordot/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp� dense_8/Tensordot/ReadVariableOp�
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_7/Tensordot/transpose	Transposeinputs!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������c
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	d
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       a
dense_8/Tensordot/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_8/Tensordot/transpose	Transposedense_7/Relu:activations:0!dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	k
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
��
�
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12157973

inputsK
5attention_query_einsum_einsum_readvariableop_resource:=
+attention_query_add_readvariableop_resource:I
3attention_key_einsum_einsum_readvariableop_resource:;
)attention_key_add_readvariableop_resource:K
5attention_value_einsum_einsum_readvariableop_resource:=
+attention_value_add_readvariableop_resource:V
@attention_attention_output_einsum_einsum_readvariableop_resource:D
6attention_attention_output_add_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:H
6sequential_1_dense_7_tensordot_readvariableop_resource:B
4sequential_1_dense_7_biasadd_readvariableop_resource:H
6sequential_1_dense_8_tensordot_readvariableop_resource:B
4sequential_1_dense_8_biasadd_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:
identity��-attention/attention_output/add/ReadVariableOp�7attention/attention_output/einsum/Einsum/ReadVariableOp� attention/key/add/ReadVariableOp�*attention/key/einsum/Einsum/ReadVariableOp�"attention/query/add/ReadVariableOp�,attention/query/einsum/Einsum/ReadVariableOp�"attention/value/add/ReadVariableOp�,attention/value/einsum/Einsum/ReadVariableOp�.layer_normalization_2/batchnorm/ReadVariableOp�2layer_normalization_2/batchnorm/mul/ReadVariableOp�.layer_normalization_3/batchnorm/ReadVariableOp�2layer_normalization_3/batchnorm/mul/ReadVariableOp�+sequential_1/dense_7/BiasAdd/ReadVariableOp�-sequential_1/dense_7/Tensordot/ReadVariableOp�+sequential_1/dense_8/BiasAdd/ReadVariableOp�-sequential_1/dense_8/Tensordot/ReadVariableOp�
,attention/query/einsum/Einsum/ReadVariableOpReadVariableOp5attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/query/einsum/EinsumEinsuminputs4attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
"attention/query/add/ReadVariableOpReadVariableOp+attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/query/addAddV2&attention/query/einsum/Einsum:output:0*attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
*attention/key/einsum/Einsum/ReadVariableOpReadVariableOp3attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/key/einsum/EinsumEinsuminputs2attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
 attention/key/add/ReadVariableOpReadVariableOp)attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/key/addAddV2$attention/key/einsum/Einsum:output:0(attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
,attention/value/einsum/Einsum/ReadVariableOpReadVariableOp5attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention/value/einsum/EinsumEinsuminputs4attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
"attention/value/add/ReadVariableOpReadVariableOp+attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
attention/value/addAddV2&attention/value/einsum/Einsum:output:0*attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	T
attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
attention/MulMulattention/query/add:z:0attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
attention/einsum/EinsumEinsumattention/key/add:z:0attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
attention/softmax/SoftmaxSoftmax attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
attention/einsum_1/EinsumEinsum#attention/softmax/Softmax:softmax:0attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
7attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp@attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
(attention/attention_output/einsum/EinsumEinsum"attention/einsum_1/Einsum:output:0?attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
-attention/attention_output/add/ReadVariableOpReadVariableOp6attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention/attention_output/addAddV21attention/attention_output/einsum/Einsum:output:05attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_6/dropout/MulMul"attention/attention_output/add:z:0 dropout_6/dropout/Const:output:0*
T0*+
_output_shapes
:���������	i
dropout_6/dropout/ShapeShape"attention/attention_output/add:z:0*
T0*
_output_shapes
:�
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	g
addAddV2inputsdropout_6/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_1/dense_7/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_7/Tensordot/GatherV2GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/free:output:05sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/dense_7/Tensordot/GatherV2_1GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/axes:output:07sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	~
sequential_1/dense_7/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
$sequential_1/dense_8/Tensordot/ShapeShape'sequential_1/dense_7/Relu:activations:0*
T0*
_output_shapes
:n
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/dense_8/Tensordot/transpose	Transpose'sequential_1/dense_7/Relu:activations:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_7/dropout/MulMul%sequential_1/dense_8/BiasAdd:output:0 dropout_7/dropout/Const:output:0*
T0*+
_output_shapes
:���������	l
dropout_7/dropout/ShapeShape%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:�
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	�
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_7/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	|
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������	: : : : : : : : : : : : : : : : 2^
-attention/attention_output/add/ReadVariableOp-attention/attention_output/add/ReadVariableOp2r
7attention/attention_output/einsum/Einsum/ReadVariableOp7attention/attention_output/einsum/Einsum/ReadVariableOp2D
 attention/key/add/ReadVariableOp attention/key/add/ReadVariableOp2X
*attention/key/einsum/Einsum/ReadVariableOp*attention/key/einsum/Einsum/ReadVariableOp2H
"attention/query/add/ReadVariableOp"attention/query/add/ReadVariableOp2\
,attention/query/einsum/Einsum/ReadVariableOp,attention/query/einsum/Einsum/ReadVariableOp2H
"attention/value/add/ReadVariableOp"attention/value/add/ReadVariableOp2\
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2^
-sequential_1/dense_7/Tensordot/ReadVariableOp-sequential_1/dense_7/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156924

inputs"
dense_7_12156913:
dense_7_12156915:"
dense_8_12156918:
dense_8_12156920:
identity��dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_12156913dense_7_12156915*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_12156821�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_12156918dense_8_12156920*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_12156857{
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
F__inference_dense_11_layer_call_and_return_conditional_losses_12159753

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
E__inference_dense_8_layer_call_and_return_conditional_losses_12156857

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������	z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
��
�?
!__inference__traced_save_12161111
file_prefix.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	T
Psavev2_token_and_position_embedding_1_embedding_2_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_1_embedding_3_embeddings_read_readvariableopI
Esavev2_transformer_block_1_attention_query_kernel_read_readvariableopG
Csavev2_transformer_block_1_attention_query_bias_read_readvariableopG
Csavev2_transformer_block_1_attention_key_kernel_read_readvariableopE
Asavev2_transformer_block_1_attention_key_bias_read_readvariableopI
Esavev2_transformer_block_1_attention_value_kernel_read_readvariableopG
Csavev2_transformer_block_1_attention_value_bias_read_readvariableopT
Psavev2_transformer_block_1_attention_attention_output_kernel_read_readvariableopR
Nsavev2_transformer_block_1_attention_attention_output_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableopN
Jsavev2_transformer_block_1_layer_normalization_2_gamma_read_readvariableopM
Isavev2_transformer_block_1_layer_normalization_2_beta_read_readvariableopN
Jsavev2_transformer_block_1_layer_normalization_3_gamma_read_readvariableopM
Isavev2_transformer_block_1_layer_normalization_3_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_1_embedding_2_embeddings_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_1_embedding_3_embeddings_m_read_readvariableopP
Lsavev2_adam_transformer_block_1_attention_query_kernel_m_read_readvariableopN
Jsavev2_adam_transformer_block_1_attention_query_bias_m_read_readvariableopN
Jsavev2_adam_transformer_block_1_attention_key_kernel_m_read_readvariableopL
Hsavev2_adam_transformer_block_1_attention_key_bias_m_read_readvariableopP
Lsavev2_adam_transformer_block_1_attention_value_kernel_m_read_readvariableopN
Jsavev2_adam_transformer_block_1_attention_value_bias_m_read_readvariableop[
Wsavev2_adam_transformer_block_1_attention_attention_output_kernel_m_read_readvariableopY
Usavev2_adam_transformer_block_1_attention_attention_output_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopU
Qsavev2_adam_transformer_block_1_layer_normalization_2_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_1_layer_normalization_2_beta_m_read_readvariableopU
Qsavev2_adam_transformer_block_1_layer_normalization_3_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_1_layer_normalization_3_beta_m_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_1_embedding_2_embeddings_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_1_embedding_3_embeddings_v_read_readvariableopP
Lsavev2_adam_transformer_block_1_attention_query_kernel_v_read_readvariableopN
Jsavev2_adam_transformer_block_1_attention_query_bias_v_read_readvariableopN
Jsavev2_adam_transformer_block_1_attention_key_kernel_v_read_readvariableopL
Hsavev2_adam_transformer_block_1_attention_key_bias_v_read_readvariableopP
Lsavev2_adam_transformer_block_1_attention_value_kernel_v_read_readvariableopN
Jsavev2_adam_transformer_block_1_attention_value_bias_v_read_readvariableop[
Wsavev2_adam_transformer_block_1_attention_attention_output_kernel_v_read_readvariableopY
Usavev2_adam_transformer_block_1_attention_attention_output_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableopU
Qsavev2_adam_transformer_block_1_layer_normalization_2_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_1_layer_normalization_2_beta_v_read_readvariableopU
Qsavev2_adam_transformer_block_1_layer_normalization_3_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_1_layer_normalization_3_beta_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:|*
dtype0*�?
value�?B�?|B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:|*
dtype0*�
value�B�|B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �=
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableopPsavev2_token_and_position_embedding_1_embedding_2_embeddings_read_readvariableopPsavev2_token_and_position_embedding_1_embedding_3_embeddings_read_readvariableopEsavev2_transformer_block_1_attention_query_kernel_read_readvariableopCsavev2_transformer_block_1_attention_query_bias_read_readvariableopCsavev2_transformer_block_1_attention_key_kernel_read_readvariableopAsavev2_transformer_block_1_attention_key_bias_read_readvariableopEsavev2_transformer_block_1_attention_value_kernel_read_readvariableopCsavev2_transformer_block_1_attention_value_bias_read_readvariableopPsavev2_transformer_block_1_attention_attention_output_kernel_read_readvariableopNsavev2_transformer_block_1_attention_attention_output_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableopJsavev2_transformer_block_1_layer_normalization_2_gamma_read_readvariableopIsavev2_transformer_block_1_layer_normalization_2_beta_read_readvariableopJsavev2_transformer_block_1_layer_normalization_3_gamma_read_readvariableopIsavev2_transformer_block_1_layer_normalization_3_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableopWsavev2_adam_token_and_position_embedding_1_embedding_2_embeddings_m_read_readvariableopWsavev2_adam_token_and_position_embedding_1_embedding_3_embeddings_m_read_readvariableopLsavev2_adam_transformer_block_1_attention_query_kernel_m_read_readvariableopJsavev2_adam_transformer_block_1_attention_query_bias_m_read_readvariableopJsavev2_adam_transformer_block_1_attention_key_kernel_m_read_readvariableopHsavev2_adam_transformer_block_1_attention_key_bias_m_read_readvariableopLsavev2_adam_transformer_block_1_attention_value_kernel_m_read_readvariableopJsavev2_adam_transformer_block_1_attention_value_bias_m_read_readvariableopWsavev2_adam_transformer_block_1_attention_attention_output_kernel_m_read_readvariableopUsavev2_adam_transformer_block_1_attention_attention_output_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableopQsavev2_adam_transformer_block_1_layer_normalization_2_gamma_m_read_readvariableopPsavev2_adam_transformer_block_1_layer_normalization_2_beta_m_read_readvariableopQsavev2_adam_transformer_block_1_layer_normalization_3_gamma_m_read_readvariableopPsavev2_adam_transformer_block_1_layer_normalization_3_beta_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopWsavev2_adam_token_and_position_embedding_1_embedding_2_embeddings_v_read_readvariableopWsavev2_adam_token_and_position_embedding_1_embedding_3_embeddings_v_read_readvariableopLsavev2_adam_transformer_block_1_attention_query_kernel_v_read_readvariableopJsavev2_adam_transformer_block_1_attention_query_bias_v_read_readvariableopJsavev2_adam_transformer_block_1_attention_key_kernel_v_read_readvariableopHsavev2_adam_transformer_block_1_attention_key_bias_v_read_readvariableopLsavev2_adam_transformer_block_1_attention_value_kernel_v_read_readvariableopJsavev2_adam_transformer_block_1_attention_value_bias_v_read_readvariableopWsavev2_adam_transformer_block_1_attention_attention_output_kernel_v_read_readvariableopUsavev2_adam_transformer_block_1_attention_attention_output_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableopQsavev2_adam_transformer_block_1_layer_normalization_2_gamma_v_read_readvariableopPsavev2_adam_transformer_block_1_layer_normalization_2_beta_v_read_readvariableopQsavev2_adam_transformer_block_1_layer_normalization_3_gamma_v_read_readvariableopPsavev2_adam_transformer_block_1_layer_normalization_3_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
~2|	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
 : : : : : :  : : : : : : : :  : : : : : : : : : :@:: : : : : ::	::::::::::::::::: : :
 : : : :  : : : : : :  : : : : : :@:::	:::::::::::::::::
 : : : :  : : : : : :  : : : : : :@:::	::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
 : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$  

_output_shapes

::$! 

_output_shapes

:	:("$
"
_output_shapes
::$# 

_output_shapes

::($$
"
_output_shapes
::$% 

_output_shapes

::(&$
"
_output_shapes
::$' 

_output_shapes

::(($
"
_output_shapes
:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::2

_output_shapes
: :3

_output_shapes
: :$4 

_output_shapes

:
 : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :$8 

_output_shapes

:  : 9

_output_shapes
: : :

_output_shapes
: : ;

_output_shapes
: :$< 

_output_shapes

: : =

_output_shapes
: :$> 

_output_shapes

:  : ?

_output_shapes
: : @

_output_shapes
: : A

_output_shapes
: : B

_output_shapes
: : C

_output_shapes
: :$D 

_output_shapes

:@: E

_output_shapes
::$F 

_output_shapes

::$G 

_output_shapes

:	:(H$
"
_output_shapes
::$I 

_output_shapes

::(J$
"
_output_shapes
::$K 

_output_shapes

::(L$
"
_output_shapes
::$M 

_output_shapes

::(N$
"
_output_shapes
:: O

_output_shapes
::$P 

_output_shapes

:: Q

_output_shapes
::$R 

_output_shapes

:: S

_output_shapes
:: T

_output_shapes
:: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
::$X 

_output_shapes

:
 : Y

_output_shapes
: : Z

_output_shapes
: : [

_output_shapes
: :$\ 

_output_shapes

:  : ]

_output_shapes
: : ^

_output_shapes
: : _

_output_shapes
: :$` 

_output_shapes

: : a

_output_shapes
: :$b 

_output_shapes

:  : c

_output_shapes
: : d

_output_shapes
: : e

_output_shapes
: : f

_output_shapes
: : g

_output_shapes
: :$h 

_output_shapes

:@: i

_output_shapes
::$j 

_output_shapes

::$k 

_output_shapes

:	:(l$
"
_output_shapes
::$m 

_output_shapes

::(n$
"
_output_shapes
::$o 

_output_shapes

::(p$
"
_output_shapes
::$q 

_output_shapes

::(r$
"
_output_shapes
:: s

_output_shapes
::$t 

_output_shapes

:: u

_output_shapes
::$v 

_output_shapes

:: w

_output_shapes
:: x

_output_shapes
:: y

_output_shapes
:: z

_output_shapes
:: {

_output_shapes
::|

_output_shapes
: 
�
�
*__inference_dense_9_layer_call_fn_12160221

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_12157531o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
Y
=__inference_global_average_pooling1d_1_layer_call_fn_12160179

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12157068i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12157142

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_dense_12_layer_call_fn_12160241

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_12157514o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_dense_10_layer_call_fn_12159581

inputs
unknown:
 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_12157255o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
E__inference_dense_9_layer_call_and_return_conditional_losses_12160232

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�2
#__inference__wrapped_model_12156701
input_3
input_4A
/model_1_dense_10_matmul_readvariableop_resource:
 >
0model_1_dense_10_biasadd_readvariableop_resource: M
?model_1_batch_normalization_5_batchnorm_readvariableop_resource: Q
Cmodel_1_batch_normalization_5_batchnorm_mul_readvariableop_resource: O
Amodel_1_batch_normalization_5_batchnorm_readvariableop_1_resource: O
Amodel_1_batch_normalization_5_batchnorm_readvariableop_2_resource: A
/model_1_dense_11_matmul_readvariableop_resource:  >
0model_1_dense_11_biasadd_readvariableop_resource: ^
Lmodel_1_token_and_position_embedding_1_embedding_3_embedding_lookup_12156489:	^
Lmodel_1_token_and_position_embedding_1_embedding_2_embedding_lookup_12156495:M
?model_1_batch_normalization_6_batchnorm_readvariableop_resource: Q
Cmodel_1_batch_normalization_6_batchnorm_mul_readvariableop_resource: O
Amodel_1_batch_normalization_6_batchnorm_readvariableop_1_resource: O
Amodel_1_batch_normalization_6_batchnorm_readvariableop_2_resource: g
Qmodel_1_transformer_block_1_attention_query_einsum_einsum_readvariableop_resource:Y
Gmodel_1_transformer_block_1_attention_query_add_readvariableop_resource:e
Omodel_1_transformer_block_1_attention_key_einsum_einsum_readvariableop_resource:W
Emodel_1_transformer_block_1_attention_key_add_readvariableop_resource:g
Qmodel_1_transformer_block_1_attention_value_einsum_einsum_readvariableop_resource:Y
Gmodel_1_transformer_block_1_attention_value_add_readvariableop_resource:r
\model_1_transformer_block_1_attention_attention_output_einsum_einsum_readvariableop_resource:`
Rmodel_1_transformer_block_1_attention_attention_output_add_readvariableop_resource:e
Wmodel_1_transformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource:a
Smodel_1_transformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource:d
Rmodel_1_transformer_block_1_sequential_1_dense_7_tensordot_readvariableop_resource:^
Pmodel_1_transformer_block_1_sequential_1_dense_7_biasadd_readvariableop_resource:d
Rmodel_1_transformer_block_1_sequential_1_dense_8_tensordot_readvariableop_resource:^
Pmodel_1_transformer_block_1_sequential_1_dense_8_biasadd_readvariableop_resource:e
Wmodel_1_transformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource:a
Smodel_1_transformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource:A
/model_1_dense_12_matmul_readvariableop_resource:  >
0model_1_dense_12_biasadd_readvariableop_resource: @
.model_1_dense_9_matmul_readvariableop_resource: =
/model_1_dense_9_biasadd_readvariableop_resource: M
?model_1_batch_normalization_7_batchnorm_readvariableop_resource: Q
Cmodel_1_batch_normalization_7_batchnorm_mul_readvariableop_resource: O
Amodel_1_batch_normalization_7_batchnorm_readvariableop_1_resource: O
Amodel_1_batch_normalization_7_batchnorm_readvariableop_2_resource: M
?model_1_batch_normalization_4_batchnorm_readvariableop_resource: Q
Cmodel_1_batch_normalization_4_batchnorm_mul_readvariableop_resource: O
Amodel_1_batch_normalization_4_batchnorm_readvariableop_1_resource: O
Amodel_1_batch_normalization_4_batchnorm_readvariableop_2_resource: A
/model_1_dense_13_matmul_readvariableop_resource:@>
0model_1_dense_13_biasadd_readvariableop_resource:
identity��6model_1/batch_normalization_4/batchnorm/ReadVariableOp�8model_1/batch_normalization_4/batchnorm/ReadVariableOp_1�8model_1/batch_normalization_4/batchnorm/ReadVariableOp_2�:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp�6model_1/batch_normalization_5/batchnorm/ReadVariableOp�8model_1/batch_normalization_5/batchnorm/ReadVariableOp_1�8model_1/batch_normalization_5/batchnorm/ReadVariableOp_2�:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp�6model_1/batch_normalization_6/batchnorm/ReadVariableOp�8model_1/batch_normalization_6/batchnorm/ReadVariableOp_1�8model_1/batch_normalization_6/batchnorm/ReadVariableOp_2�:model_1/batch_normalization_6/batchnorm/mul/ReadVariableOp�6model_1/batch_normalization_7/batchnorm/ReadVariableOp�8model_1/batch_normalization_7/batchnorm/ReadVariableOp_1�8model_1/batch_normalization_7/batchnorm/ReadVariableOp_2�:model_1/batch_normalization_7/batchnorm/mul/ReadVariableOp�'model_1/dense_10/BiasAdd/ReadVariableOp�&model_1/dense_10/MatMul/ReadVariableOp�'model_1/dense_11/BiasAdd/ReadVariableOp�&model_1/dense_11/MatMul/ReadVariableOp�'model_1/dense_12/BiasAdd/ReadVariableOp�&model_1/dense_12/MatMul/ReadVariableOp�'model_1/dense_13/BiasAdd/ReadVariableOp�&model_1/dense_13/MatMul/ReadVariableOp�&model_1/dense_9/BiasAdd/ReadVariableOp�%model_1/dense_9/MatMul/ReadVariableOp�Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup�Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup�Imodel_1/transformer_block_1/attention/attention_output/add/ReadVariableOp�Smodel_1/transformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp�<model_1/transformer_block_1/attention/key/add/ReadVariableOp�Fmodel_1/transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp�>model_1/transformer_block_1/attention/query/add/ReadVariableOp�Hmodel_1/transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp�>model_1/transformer_block_1/attention/value/add/ReadVariableOp�Hmodel_1/transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp�Jmodel_1/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp�Nmodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp�Jmodel_1/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp�Nmodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp�Gmodel_1/transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp�Imodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp�Gmodel_1/transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp�Imodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp�
&model_1/dense_10/MatMul/ReadVariableOpReadVariableOp/model_1_dense_10_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype0�
model_1/dense_10/MatMulMatMulinput_4.model_1/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_10/BiasAddBiasAdd!model_1/dense_10/MatMul:product:0/model_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
model_1/dense_10/ReluRelu!model_1/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
6model_1/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0r
-model_1/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_1/batch_normalization_5/batchnorm/addAddV2>model_1/batch_normalization_5/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_5/batchnorm/RsqrtRsqrt/model_1/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: �
:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
+model_1/batch_normalization_5/batchnorm/mulMul1model_1/batch_normalization_5/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_5/batchnorm/mul_1Mul#model_1/dense_10/Relu:activations:0/model_1/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
-model_1/batch_normalization_5/batchnorm/mul_2Mul@model_1/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: �
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
+model_1/batch_normalization_5/batchnorm/subSub@model_1/batch_normalization_5/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_5/batchnorm/add_1AddV21model_1/batch_normalization_5/batchnorm/mul_1:z:0/model_1/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
model_1/dropout_9/IdentityIdentity1model_1/batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� �
&model_1/dense_11/MatMul/ReadVariableOpReadVariableOp/model_1_dense_11_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_1/dense_11/MatMulMatMul#model_1/dropout_9/Identity:output:0.model_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_11/BiasAddBiasAdd!model_1/dense_11/MatMul:product:0/model_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
model_1/dense_11/ReluRelu!model_1/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:��������� c
,model_1/token_and_position_embedding_1/ShapeShapeinput_3*
T0*
_output_shapes
:�
:model_1/token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
<model_1/token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
<model_1/token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4model_1/token_and_position_embedding_1/strided_sliceStridedSlice5model_1/token_and_position_embedding_1/Shape:output:0Cmodel_1/token_and_position_embedding_1/strided_slice/stack:output:0Emodel_1/token_and_position_embedding_1/strided_slice/stack_1:output:0Emodel_1/token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2model_1/token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : t
2model_1/token_and_position_embedding_1/range/limitConst*
_output_shapes
: *
dtype0*
value	B :	t
2model_1/token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
,model_1/token_and_position_embedding_1/rangeRange;model_1/token_and_position_embedding_1/range/start:output:0;model_1/token_and_position_embedding_1/range/limit:output:0;model_1/token_and_position_embedding_1/range/delta:output:0*
_output_shapes
:	�
Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherLmodel_1_token_and_position_embedding_1_embedding_3_embedding_lookup_121564895model_1/token_and_position_embedding_1/range:output:0*
Tindices0*_
_classU
SQloc:@model_1/token_and_position_embedding_1/embedding_3/embedding_lookup/12156489*
_output_shapes

:	*
dtype0�
Lmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityLmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup:output:0*
T0*_
_classU
SQloc:@model_1/token_and_position_embedding_1/embedding_3/embedding_lookup/12156489*
_output_shapes

:	�
Nmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityUmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	�
7model_1/token_and_position_embedding_1/embedding_2/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:���������	�
Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherLmodel_1_token_and_position_embedding_1_embedding_2_embedding_lookup_12156495;model_1/token_and_position_embedding_1/embedding_2/Cast:y:0*
Tindices0*_
_classU
SQloc:@model_1/token_and_position_embedding_1/embedding_2/embedding_lookup/12156495*+
_output_shapes
:���������	*
dtype0�
Lmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityLmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup:output:0*
T0*_
_classU
SQloc:@model_1/token_and_position_embedding_1/embedding_2/embedding_lookup/12156495*+
_output_shapes
:���������	�
Nmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityUmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
*model_1/token_and_position_embedding_1/addAddV2Wmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Wmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	�
6model_1/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0r
-model_1/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_1/batch_normalization_6/batchnorm/addAddV2>model_1/batch_normalization_6/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_6/batchnorm/RsqrtRsqrt/model_1/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: �
:model_1/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
+model_1/batch_normalization_6/batchnorm/mulMul1model_1/batch_normalization_6/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_6/batchnorm/mul_1Mul#model_1/dense_11/Relu:activations:0/model_1/batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
8model_1/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
-model_1/batch_normalization_6/batchnorm/mul_2Mul@model_1/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: �
8model_1/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
+model_1/batch_normalization_6/batchnorm/subSub@model_1/batch_normalization_6/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_6/batchnorm/add_1AddV21model_1/batch_normalization_6/batchnorm/mul_1:z:0/model_1/batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
Hmodel_1/transformer_block_1/attention/query/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_1_transformer_block_1_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
9model_1/transformer_block_1/attention/query/einsum/EinsumEinsum.model_1/token_and_position_embedding_1/add:z:0Pmodel_1/transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
>model_1/transformer_block_1/attention/query/add/ReadVariableOpReadVariableOpGmodel_1_transformer_block_1_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
/model_1/transformer_block_1/attention/query/addAddV2Bmodel_1/transformer_block_1/attention/query/einsum/Einsum:output:0Fmodel_1/transformer_block_1/attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
Fmodel_1/transformer_block_1/attention/key/einsum/Einsum/ReadVariableOpReadVariableOpOmodel_1_transformer_block_1_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
7model_1/transformer_block_1/attention/key/einsum/EinsumEinsum.model_1/token_and_position_embedding_1/add:z:0Nmodel_1/transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
<model_1/transformer_block_1/attention/key/add/ReadVariableOpReadVariableOpEmodel_1_transformer_block_1_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
-model_1/transformer_block_1/attention/key/addAddV2@model_1/transformer_block_1/attention/key/einsum/Einsum:output:0Dmodel_1/transformer_block_1/attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
Hmodel_1/transformer_block_1/attention/value/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_1_transformer_block_1_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
9model_1/transformer_block_1/attention/value/einsum/EinsumEinsum.model_1/token_and_position_embedding_1/add:z:0Pmodel_1/transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
>model_1/transformer_block_1/attention/value/add/ReadVariableOpReadVariableOpGmodel_1_transformer_block_1_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
/model_1/transformer_block_1/attention/value/addAddV2Bmodel_1/transformer_block_1/attention/value/einsum/Einsum:output:0Fmodel_1/transformer_block_1/attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	p
+model_1/transformer_block_1/attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
)model_1/transformer_block_1/attention/MulMul3model_1/transformer_block_1/attention/query/add:z:04model_1/transformer_block_1/attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
3model_1/transformer_block_1/attention/einsum/EinsumEinsum1model_1/transformer_block_1/attention/key/add:z:0-model_1/transformer_block_1/attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
5model_1/transformer_block_1/attention/softmax/SoftmaxSoftmax<model_1/transformer_block_1/attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
6model_1/transformer_block_1/attention/dropout/IdentityIdentity?model_1/transformer_block_1/attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������		�
5model_1/transformer_block_1/attention/einsum_1/EinsumEinsum?model_1/transformer_block_1/attention/dropout/Identity:output:03model_1/transformer_block_1/attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
Smodel_1/transformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp\model_1_transformer_block_1_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Dmodel_1/transformer_block_1/attention/attention_output/einsum/EinsumEinsum>model_1/transformer_block_1/attention/einsum_1/Einsum:output:0[model_1/transformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
Imodel_1/transformer_block_1/attention/attention_output/add/ReadVariableOpReadVariableOpRmodel_1_transformer_block_1_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
:model_1/transformer_block_1/attention/attention_output/addAddV2Mmodel_1/transformer_block_1/attention/attention_output/einsum/Einsum:output:0Qmodel_1/transformer_block_1/attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
.model_1/transformer_block_1/dropout_6/IdentityIdentity>model_1/transformer_block_1/attention/attention_output/add:z:0*
T0*+
_output_shapes
:���������	�
model_1/transformer_block_1/addAddV2.model_1/token_and_position_embedding_1/add:z:07model_1/transformer_block_1/dropout_6/Identity:output:0*
T0*+
_output_shapes
:���������	�
Pmodel_1/transformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>model_1/transformer_block_1/layer_normalization_2/moments/meanMean#model_1/transformer_block_1/add:z:0Ymodel_1/transformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
Fmodel_1/transformer_block_1/layer_normalization_2/moments/StopGradientStopGradientGmodel_1/transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Kmodel_1/transformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifference#model_1/transformer_block_1/add:z:0Omodel_1/transformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Tmodel_1/transformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Bmodel_1/transformer_block_1/layer_normalization_2/moments/varianceMeanOmodel_1/transformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0]model_1/transformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
Amodel_1/transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
?model_1/transformer_block_1/layer_normalization_2/batchnorm/addAddV2Kmodel_1/transformer_block_1/layer_normalization_2/moments/variance:output:0Jmodel_1/transformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
Amodel_1/transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrtCmodel_1/transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Nmodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_1_transformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
?model_1/transformer_block_1/layer_normalization_2/batchnorm/mulMulEmodel_1/transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Vmodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
Amodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul_1Mul#model_1/transformer_block_1/add:z:0Cmodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Amodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul_2MulGmodel_1/transformer_block_1/layer_normalization_2/moments/mean:output:0Cmodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Jmodel_1/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpSmodel_1_transformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
?model_1/transformer_block_1/layer_normalization_2/batchnorm/subSubRmodel_1/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0Emodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
Amodel_1/transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2Emodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0Cmodel_1/transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
Imodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOpRmodel_1_transformer_block_1_sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
?model_1/transformer_block_1/sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
?model_1/transformer_block_1/sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
@model_1/transformer_block_1/sequential_1/dense_7/Tensordot/ShapeShapeEmodel_1/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Hmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Cmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2GatherV2Imodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Shape:output:0Hmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/free:output:0Qmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Emodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1GatherV2Imodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Shape:output:0Hmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/axes:output:0Smodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@model_1/transformer_block_1/sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
?model_1/transformer_block_1/sequential_1/dense_7/Tensordot/ProdProdLmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2:output:0Imodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Bmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Amodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Prod_1ProdNmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2_1:output:0Kmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Fmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Amodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/concatConcatV2Hmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/free:output:0Hmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/axes:output:0Omodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
@model_1/transformer_block_1/sequential_1/dense_7/Tensordot/stackPackHmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Prod:output:0Jmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/transpose	TransposeEmodel_1/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Jmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Bmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/ReshapeReshapeHmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/transpose:y:0Imodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Amodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/MatMulMatMulKmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Reshape:output:0Qmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Bmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Cmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/concat_1ConcatV2Lmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/GatherV2:output:0Kmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/Const_2:output:0Qmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
:model_1/transformer_block_1/sequential_1/dense_7/TensordotReshapeKmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/MatMul:product:0Lmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
Gmodel_1/transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOpPmodel_1_transformer_block_1_sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
8model_1/transformer_block_1/sequential_1/dense_7/BiasAddBiasAddCmodel_1/transformer_block_1/sequential_1/dense_7/Tensordot:output:0Omodel_1/transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
5model_1/transformer_block_1/sequential_1/dense_7/ReluReluAmodel_1/transformer_block_1/sequential_1/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
Imodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOpRmodel_1_transformer_block_1_sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
?model_1/transformer_block_1/sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
?model_1/transformer_block_1/sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
@model_1/transformer_block_1/sequential_1/dense_8/Tensordot/ShapeShapeCmodel_1/transformer_block_1/sequential_1/dense_7/Relu:activations:0*
T0*
_output_shapes
:�
Hmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Cmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2GatherV2Imodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Shape:output:0Hmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/free:output:0Qmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Emodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1GatherV2Imodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Shape:output:0Hmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/axes:output:0Smodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@model_1/transformer_block_1/sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
?model_1/transformer_block_1/sequential_1/dense_8/Tensordot/ProdProdLmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2:output:0Imodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Bmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Amodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Prod_1ProdNmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2_1:output:0Kmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Fmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Amodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/concatConcatV2Hmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/free:output:0Hmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/axes:output:0Omodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
@model_1/transformer_block_1/sequential_1/dense_8/Tensordot/stackPackHmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Prod:output:0Jmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/transpose	TransposeCmodel_1/transformer_block_1/sequential_1/dense_7/Relu:activations:0Jmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Bmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/ReshapeReshapeHmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/transpose:y:0Imodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Amodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/MatMulMatMulKmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Reshape:output:0Qmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Bmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Cmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/concat_1ConcatV2Lmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/GatherV2:output:0Kmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/Const_2:output:0Qmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
:model_1/transformer_block_1/sequential_1/dense_8/TensordotReshapeKmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/MatMul:product:0Lmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
Gmodel_1/transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOpPmodel_1_transformer_block_1_sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
8model_1/transformer_block_1/sequential_1/dense_8/BiasAddBiasAddCmodel_1/transformer_block_1/sequential_1/dense_8/Tensordot:output:0Omodel_1/transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
.model_1/transformer_block_1/dropout_7/IdentityIdentityAmodel_1/transformer_block_1/sequential_1/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
!model_1/transformer_block_1/add_1AddV2Emodel_1/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:07model_1/transformer_block_1/dropout_7/Identity:output:0*
T0*+
_output_shapes
:���������	�
Pmodel_1/transformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>model_1/transformer_block_1/layer_normalization_3/moments/meanMean%model_1/transformer_block_1/add_1:z:0Ymodel_1/transformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
Fmodel_1/transformer_block_1/layer_normalization_3/moments/StopGradientStopGradientGmodel_1/transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Kmodel_1/transformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifference%model_1/transformer_block_1/add_1:z:0Omodel_1/transformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Tmodel_1/transformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Bmodel_1/transformer_block_1/layer_normalization_3/moments/varianceMeanOmodel_1/transformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0]model_1/transformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
Amodel_1/transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
?model_1/transformer_block_1/layer_normalization_3/batchnorm/addAddV2Kmodel_1/transformer_block_1/layer_normalization_3/moments/variance:output:0Jmodel_1/transformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
Amodel_1/transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrtCmodel_1/transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Nmodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_1_transformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
?model_1/transformer_block_1/layer_normalization_3/batchnorm/mulMulEmodel_1/transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Vmodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
Amodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul_1Mul%model_1/transformer_block_1/add_1:z:0Cmodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Amodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul_2MulGmodel_1/transformer_block_1/layer_normalization_3/moments/mean:output:0Cmodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Jmodel_1/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpSmodel_1_transformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
?model_1/transformer_block_1/layer_normalization_3/batchnorm/subSubRmodel_1/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0Emodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
Amodel_1/transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2Emodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0Cmodel_1/transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
model_1/dropout_10/IdentityIdentity1model_1/batch_normalization_6/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� {
9model_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'model_1/global_average_pooling1d_1/MeanMeanEmodel_1/transformer_block_1/layer_normalization_3/batchnorm/add_1:z:0Bmodel_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
&model_1/dense_12/MatMul/ReadVariableOpReadVariableOp/model_1_dense_12_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_1/dense_12/MatMulMatMul$model_1/dropout_10/Identity:output:0.model_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_12/BiasAddBiasAdd!model_1/dense_12/MatMul:product:0/model_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
model_1/dense_12/ReluRelu!model_1/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
%model_1/dense_9/MatMul/ReadVariableOpReadVariableOp.model_1_dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
model_1/dense_9/MatMulMatMul0model_1/global_average_pooling1d_1/Mean:output:0-model_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_9/BiasAddBiasAdd model_1/dense_9/MatMul:product:0.model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
model_1/dense_9/ReluRelu model_1/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
6model_1/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0r
-model_1/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_1/batch_normalization_7/batchnorm/addAddV2>model_1/batch_normalization_7/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_7/batchnorm/RsqrtRsqrt/model_1/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: �
:model_1/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
+model_1/batch_normalization_7/batchnorm/mulMul1model_1/batch_normalization_7/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_7/batchnorm/mul_1Mul#model_1/dense_12/Relu:activations:0/model_1/batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
8model_1/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
-model_1/batch_normalization_7/batchnorm/mul_2Mul@model_1/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: �
8model_1/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
+model_1/batch_normalization_7/batchnorm/subSub@model_1/batch_normalization_7/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_7/batchnorm/add_1AddV21model_1/batch_normalization_7/batchnorm/mul_1:z:0/model_1/batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
6model_1/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0r
-model_1/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_1/batch_normalization_4/batchnorm/addAddV2>model_1/batch_normalization_4/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_4/batchnorm/RsqrtRsqrt/model_1/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: �
:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
+model_1/batch_normalization_4/batchnorm/mulMul1model_1/batch_normalization_4/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_4/batchnorm/mul_1Mul"model_1/dense_9/Relu:activations:0/model_1/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
-model_1/batch_normalization_4/batchnorm/mul_2Mul@model_1/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: �
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
+model_1/batch_normalization_4/batchnorm/subSub@model_1/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
-model_1/batch_normalization_4/batchnorm/add_1AddV21model_1/batch_normalization_4/batchnorm/mul_1:z:0/model_1/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
model_1/dropout_8/IdentityIdentity1model_1/batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� �
model_1/dropout_11/IdentityIdentity1model_1/batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� c
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/concatenate_1/concatConcatV2#model_1/dropout_8/Identity:output:0$model_1/dropout_11/Identity:output:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
&model_1/dense_13/MatMul/ReadVariableOpReadVariableOp/model_1_dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_1/dense_13/MatMulMatMul%model_1/concatenate_1/concat:output:0.model_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/dense_13/BiasAddBiasAdd!model_1/dense_13/MatMul:product:0/model_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
model_1/dense_13/SoftmaxSoftmax!model_1/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"model_1/dense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp7^model_1/batch_normalization_4/batchnorm/ReadVariableOp9^model_1/batch_normalization_4/batchnorm/ReadVariableOp_19^model_1/batch_normalization_4/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_5/batchnorm/ReadVariableOp9^model_1/batch_normalization_5/batchnorm/ReadVariableOp_19^model_1/batch_normalization_5/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_6/batchnorm/ReadVariableOp9^model_1/batch_normalization_6/batchnorm/ReadVariableOp_19^model_1/batch_normalization_6/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_6/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_7/batchnorm/ReadVariableOp9^model_1/batch_normalization_7/batchnorm/ReadVariableOp_19^model_1/batch_normalization_7/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_7/batchnorm/mul/ReadVariableOp(^model_1/dense_10/BiasAdd/ReadVariableOp'^model_1/dense_10/MatMul/ReadVariableOp(^model_1/dense_11/BiasAdd/ReadVariableOp'^model_1/dense_11/MatMul/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp'^model_1/dense_9/BiasAdd/ReadVariableOp&^model_1/dense_9/MatMul/ReadVariableOpD^model_1/token_and_position_embedding_1/embedding_2/embedding_lookupD^model_1/token_and_position_embedding_1/embedding_3/embedding_lookupJ^model_1/transformer_block_1/attention/attention_output/add/ReadVariableOpT^model_1/transformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp=^model_1/transformer_block_1/attention/key/add/ReadVariableOpG^model_1/transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp?^model_1/transformer_block_1/attention/query/add/ReadVariableOpI^model_1/transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp?^model_1/transformer_block_1/attention/value/add/ReadVariableOpI^model_1/transformer_block_1/attention/value/einsum/Einsum/ReadVariableOpK^model_1/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpO^model_1/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpK^model_1/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpO^model_1/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpH^model_1/transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOpJ^model_1/transformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOpH^model_1/transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOpJ^model_1/transformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6model_1/batch_normalization_4/batchnorm/ReadVariableOp6model_1/batch_normalization_4/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_18model_1/batch_normalization_4/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_28model_1/batch_normalization_4/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp2p
6model_1/batch_normalization_5/batchnorm/ReadVariableOp6model_1/batch_normalization_5/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_18model_1/batch_normalization_5/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_28model_1/batch_normalization_5/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp2p
6model_1/batch_normalization_6/batchnorm/ReadVariableOp6model_1/batch_normalization_6/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_6/batchnorm/ReadVariableOp_18model_1/batch_normalization_6/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_6/batchnorm/ReadVariableOp_28model_1/batch_normalization_6/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_6/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_6/batchnorm/mul/ReadVariableOp2p
6model_1/batch_normalization_7/batchnorm/ReadVariableOp6model_1/batch_normalization_7/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_7/batchnorm/ReadVariableOp_18model_1/batch_normalization_7/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_7/batchnorm/ReadVariableOp_28model_1/batch_normalization_7/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_7/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_7/batchnorm/mul/ReadVariableOp2R
'model_1/dense_10/BiasAdd/ReadVariableOp'model_1/dense_10/BiasAdd/ReadVariableOp2P
&model_1/dense_10/MatMul/ReadVariableOp&model_1/dense_10/MatMul/ReadVariableOp2R
'model_1/dense_11/BiasAdd/ReadVariableOp'model_1/dense_11/BiasAdd/ReadVariableOp2P
&model_1/dense_11/MatMul/ReadVariableOp&model_1/dense_11/MatMul/ReadVariableOp2R
'model_1/dense_12/BiasAdd/ReadVariableOp'model_1/dense_12/BiasAdd/ReadVariableOp2P
&model_1/dense_12/MatMul/ReadVariableOp&model_1/dense_12/MatMul/ReadVariableOp2R
'model_1/dense_13/BiasAdd/ReadVariableOp'model_1/dense_13/BiasAdd/ReadVariableOp2P
&model_1/dense_13/MatMul/ReadVariableOp&model_1/dense_13/MatMul/ReadVariableOp2P
&model_1/dense_9/BiasAdd/ReadVariableOp&model_1/dense_9/BiasAdd/ReadVariableOp2N
%model_1/dense_9/MatMul/ReadVariableOp%model_1/dense_9/MatMul/ReadVariableOp2�
Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookupCmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup2�
Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookupCmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup2�
Imodel_1/transformer_block_1/attention/attention_output/add/ReadVariableOpImodel_1/transformer_block_1/attention/attention_output/add/ReadVariableOp2�
Smodel_1/transformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOpSmodel_1/transformer_block_1/attention/attention_output/einsum/Einsum/ReadVariableOp2|
<model_1/transformer_block_1/attention/key/add/ReadVariableOp<model_1/transformer_block_1/attention/key/add/ReadVariableOp2�
Fmodel_1/transformer_block_1/attention/key/einsum/Einsum/ReadVariableOpFmodel_1/transformer_block_1/attention/key/einsum/Einsum/ReadVariableOp2�
>model_1/transformer_block_1/attention/query/add/ReadVariableOp>model_1/transformer_block_1/attention/query/add/ReadVariableOp2�
Hmodel_1/transformer_block_1/attention/query/einsum/Einsum/ReadVariableOpHmodel_1/transformer_block_1/attention/query/einsum/Einsum/ReadVariableOp2�
>model_1/transformer_block_1/attention/value/add/ReadVariableOp>model_1/transformer_block_1/attention/value/add/ReadVariableOp2�
Hmodel_1/transformer_block_1/attention/value/einsum/Einsum/ReadVariableOpHmodel_1/transformer_block_1/attention/value/einsum/Einsum/ReadVariableOp2�
Jmodel_1/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpJmodel_1/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2�
Nmodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpNmodel_1/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2�
Jmodel_1/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpJmodel_1/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2�
Nmodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpNmodel_1/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2�
Gmodel_1/transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOpGmodel_1/transformer_block_1/sequential_1/dense_7/BiasAdd/ReadVariableOp2�
Imodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOpImodel_1/transformer_block_1/sequential_1/dense_7/Tensordot/ReadVariableOp2�
Gmodel_1/transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOpGmodel_1/transformer_block_1/sequential_1/dense_8/BiasAdd/ReadVariableOp2�
Imodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOpImodel_1/transformer_block_1/sequential_1/dense_8/Tensordot/ReadVariableOp:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������

!
_user_specified_name	input_4
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156976
dense_7_input"
dense_7_12156965:
dense_7_12156967:"
dense_8_12156970:
dense_8_12156972:
identity��dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_12156965dense_7_12156967*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_12156821�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_12156970dense_8_12156972*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_12156857{
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Z V
+
_output_shapes
:���������	
'
_user_specified_namedense_7_input
�
e
G__inference_dropout_9_layer_call_and_return_conditional_losses_12157275

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_dense_13_layer_call_fn_12160488

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_12157589o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12160140

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12159672

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12156725

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12160412

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12157224

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_30
serving_default_input_3:0���������	
;
input_40
serving_default_input_4:0���������
<
dense_130
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
�
%axis
	&gamma
'beta
(moving_mean
)moving_variance
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4_random_generator
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
�
7	token_emb
8pos_emb
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
�

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Gatt
Hffn
I
layernorm1
J
layernorm2
Kdropout1
Ldropout2
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h_random_generator
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kkernel
lbias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
�

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
{axis
	|gamma
}beta
~moving_mean
moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�beta_1
�beta_2

�decay
�learning_rate
	�iterm�m�&m�'m�?m�@m�Tm�Um�km�lm�sm�tm�|m�}m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�v�v�&v�'v�?v�@v�Tv�Uv�kv�lv�sv�tv�|v�}v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
0
1
&2
'3
(4
)5
�6
�7
?8
@9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
T26
U27
V28
W29
k30
l31
s32
t33
|34
}35
~36
37
�38
�39
�40
�41
�42
�43"
trackable_list_wrapper
�
0
1
&2
'3
�4
�5
?6
@7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
T24
U25
k26
l27
s28
t29
|30
}31
�32
�33
�34
�35"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_model_1_layer_call_fn_12157687
*__inference_model_1_layer_call_fn_12158773
*__inference_model_1_layer_call_fn_12158867
*__inference_model_1_layer_call_fn_12158453�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_model_1_layer_call_and_return_conditional_losses_12159123
E__inference_model_1_layer_call_and_return_conditional_losses_12159476
E__inference_model_1_layer_call_and_return_conditional_losses_12158563
E__inference_model_1_layer_call_and_return_conditional_losses_12158673�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
#__inference__wrapped_model_12156701input_3input_4"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
!:
 2dense_10/kernel
: 2dense_10/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_10_layer_call_fn_12159581�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_10_layer_call_and_return_conditional_losses_12159592�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
):' 2batch_normalization_5/gamma
(:& 2batch_normalization_5/beta
1:/  (2!batch_normalization_5/moving_mean
5:3  (2%batch_normalization_5/moving_variance
<
&0
'1
(2
)3"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_batch_normalization_5_layer_call_fn_12159605
8__inference_batch_normalization_5_layer_call_fn_12159618�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12159638
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12159672�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
,__inference_dropout_9_layer_call_fn_12159677
,__inference_dropout_9_layer_call_fn_12159682�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_9_layer_call_and_return_conditional_losses_12159687
G__inference_dropout_9_layer_call_and_return_conditional_losses_12159699�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�
�
embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�
embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�2�
A__inference_token_and_position_embedding_1_layer_call_fn_12159708�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12159733�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!:  2dense_11/kernel
: 2dense_11/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_11_layer_call_fn_12159742�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_11_layer_call_and_return_conditional_losses_12159753�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
	�axis

�gamma
	�beta
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�axis

�gamma
	�beta
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�2�
6__inference_transformer_block_1_layer_call_fn_12159790
6__inference_transformer_block_1_layer_call_fn_12159827�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12159954
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12160094�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
<
T0
U1
V2
W3"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_batch_normalization_6_layer_call_fn_12160107
8__inference_batch_normalization_6_layer_call_fn_12160120�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12160140
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12160174�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�2�
=__inference_global_average_pooling1d_1_layer_call_fn_12160179�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12160185�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
-__inference_dropout_10_layer_call_fn_12160190
-__inference_dropout_10_layer_call_fn_12160195�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_dropout_10_layer_call_and_return_conditional_losses_12160200
H__inference_dropout_10_layer_call_and_return_conditional_losses_12160212�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 : 2dense_9/kernel
: 2dense_9/bias
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_dense_9_layer_call_fn_12160221�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_9_layer_call_and_return_conditional_losses_12160232�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!:  2dense_12/kernel
: 2dense_12/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_12_layer_call_fn_12160241�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_12_layer_call_and_return_conditional_losses_12160252�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
):' 2batch_normalization_4/gamma
(:& 2batch_normalization_4/beta
1:/  (2!batch_normalization_4/moving_mean
5:3  (2%batch_normalization_4/moving_variance
<
|0
}1
~2
3"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_batch_normalization_4_layer_call_fn_12160265
8__inference_batch_normalization_4_layer_call_fn_12160278�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12160298
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12160332�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_batch_normalization_7_layer_call_fn_12160345
8__inference_batch_normalization_7_layer_call_fn_12160358�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12160378
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12160412�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
,__inference_dropout_8_layer_call_fn_12160417
,__inference_dropout_8_layer_call_fn_12160422�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_8_layer_call_and_return_conditional_losses_12160427
G__inference_dropout_8_layer_call_and_return_conditional_losses_12160439�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
-__inference_dropout_11_layer_call_fn_12160444
-__inference_dropout_11_layer_call_fn_12160449�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_dropout_11_layer_call_and_return_conditional_losses_12160454
H__inference_dropout_11_layer_call_and_return_conditional_losses_12160466�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
0__inference_concatenate_1_layer_call_fn_12160472�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12160479�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!:@2dense_13/kernel
:2dense_13/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_13_layer_call_fn_12160488�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_13_layer_call_and_return_conditional_losses_12160499�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
G:E25token_and_position_embedding_1/embedding_2/embeddings
G:E	25token_and_position_embedding_1/embedding_3/embeddings
@:>2*transformer_block_1/attention/query/kernel
::82(transformer_block_1/attention/query/bias
>:<2(transformer_block_1/attention/key/kernel
8:62&transformer_block_1/attention/key/bias
@:>2*transformer_block_1/attention/value/kernel
::82(transformer_block_1/attention/value/bias
K:I25transformer_block_1/attention/attention_output/kernel
A:?23transformer_block_1/attention/attention_output/bias
 :2dense_7/kernel
:2dense_7/bias
 :2dense_8/kernel
:2dense_8/bias
=:;2/transformer_block_1/layer_normalization_2/gamma
<::2.transformer_block_1/layer_normalization_2/beta
=:;2/transformer_block_1/layer_normalization_3/gamma
<::2.transformer_block_1/layer_normalization_3/beta
Z
(0
)1
V2
W3
~4
5
�6
�7"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_signature_wrapper_12159572input_3input_4"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpece
args]�Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpece
args]�Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_sequential_1_layer_call_fn_12156875
/__inference_sequential_1_layer_call_fn_12160512
/__inference_sequential_1_layer_call_fn_12160525
/__inference_sequential_1_layer_call_fn_12156948�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_sequential_1_layer_call_and_return_conditional_losses_12160582
J__inference_sequential_1_layer_call_and_return_conditional_losses_12160639
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156962
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156976�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
J
G0
H1
I2
J3
K4
L5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_dense_7_layer_call_fn_12160648�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_7_layer_call_and_return_conditional_losses_12160679�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_dense_8_layer_call_fn_12160688�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_8_layer_call_and_return_conditional_losses_12160718�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
&:$
 2Adam/dense_10/kernel/m
 : 2Adam/dense_10/bias/m
.:, 2"Adam/batch_normalization_5/gamma/m
-:+ 2!Adam/batch_normalization_5/beta/m
&:$  2Adam/dense_11/kernel/m
 : 2Adam/dense_11/bias/m
.:, 2"Adam/batch_normalization_6/gamma/m
-:+ 2!Adam/batch_normalization_6/beta/m
%:# 2Adam/dense_9/kernel/m
: 2Adam/dense_9/bias/m
&:$  2Adam/dense_12/kernel/m
 : 2Adam/dense_12/bias/m
.:, 2"Adam/batch_normalization_4/gamma/m
-:+ 2!Adam/batch_normalization_4/beta/m
.:, 2"Adam/batch_normalization_7/gamma/m
-:+ 2!Adam/batch_normalization_7/beta/m
&:$@2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
L:J2<Adam/token_and_position_embedding_1/embedding_2/embeddings/m
L:J	2<Adam/token_and_position_embedding_1/embedding_3/embeddings/m
E:C21Adam/transformer_block_1/attention/query/kernel/m
?:=2/Adam/transformer_block_1/attention/query/bias/m
C:A2/Adam/transformer_block_1/attention/key/kernel/m
=:;2-Adam/transformer_block_1/attention/key/bias/m
E:C21Adam/transformer_block_1/attention/value/kernel/m
?:=2/Adam/transformer_block_1/attention/value/bias/m
P:N2<Adam/transformer_block_1/attention/attention_output/kernel/m
F:D2:Adam/transformer_block_1/attention/attention_output/bias/m
%:#2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
%:#2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
B:@26Adam/transformer_block_1/layer_normalization_2/gamma/m
A:?25Adam/transformer_block_1/layer_normalization_2/beta/m
B:@26Adam/transformer_block_1/layer_normalization_3/gamma/m
A:?25Adam/transformer_block_1/layer_normalization_3/beta/m
&:$
 2Adam/dense_10/kernel/v
 : 2Adam/dense_10/bias/v
.:, 2"Adam/batch_normalization_5/gamma/v
-:+ 2!Adam/batch_normalization_5/beta/v
&:$  2Adam/dense_11/kernel/v
 : 2Adam/dense_11/bias/v
.:, 2"Adam/batch_normalization_6/gamma/v
-:+ 2!Adam/batch_normalization_6/beta/v
%:# 2Adam/dense_9/kernel/v
: 2Adam/dense_9/bias/v
&:$  2Adam/dense_12/kernel/v
 : 2Adam/dense_12/bias/v
.:, 2"Adam/batch_normalization_4/gamma/v
-:+ 2!Adam/batch_normalization_4/beta/v
.:, 2"Adam/batch_normalization_7/gamma/v
-:+ 2!Adam/batch_normalization_7/beta/v
&:$@2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
L:J2<Adam/token_and_position_embedding_1/embedding_2/embeddings/v
L:J	2<Adam/token_and_position_embedding_1/embedding_3/embeddings/v
E:C21Adam/transformer_block_1/attention/query/kernel/v
?:=2/Adam/transformer_block_1/attention/query/bias/v
C:A2/Adam/transformer_block_1/attention/key/kernel/v
=:;2-Adam/transformer_block_1/attention/key/bias/v
E:C21Adam/transformer_block_1/attention/value/kernel/v
?:=2/Adam/transformer_block_1/attention/value/bias/v
P:N2<Adam/transformer_block_1/attention/attention_output/kernel/v
F:D2:Adam/transformer_block_1/attention/attention_output/bias/v
%:#2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
%:#2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
B:@26Adam/transformer_block_1/layer_normalization_2/gamma/v
A:?25Adam/transformer_block_1/layer_normalization_2/beta/v
B:@26Adam/transformer_block_1/layer_normalization_3/gamma/v
A:?25Adam/transformer_block_1/layer_normalization_3/beta/v�
#__inference__wrapped_model_12156701�D)&('?@��WTVU����������������stkl����|~}��X�U
N�K
I�F
!�
input_3���������	
!�
input_4���������

� "3�0
.
dense_13"�
dense_13����������
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12160298b|~}3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12160332b~|}3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
8__inference_batch_normalization_4_layer_call_fn_12160265U|~}3�0
)�&
 �
inputs��������� 
p 
� "���������� �
8__inference_batch_normalization_4_layer_call_fn_12160278U~|}3�0
)�&
 �
inputs��������� 
p
� "���������� �
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12159638b)&('3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_12159672b()&'3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
8__inference_batch_normalization_5_layer_call_fn_12159605U)&('3�0
)�&
 �
inputs��������� 
p 
� "���������� �
8__inference_batch_normalization_5_layer_call_fn_12159618U()&'3�0
)�&
 �
inputs��������� 
p
� "���������� �
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12160140bWTVU3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_12160174bVWTU3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
8__inference_batch_normalization_6_layer_call_fn_12160107UWTVU3�0
)�&
 �
inputs��������� 
p 
� "���������� �
8__inference_batch_normalization_6_layer_call_fn_12160120UVWTU3�0
)�&
 �
inputs��������� 
p
� "���������� �
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12160378f����3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12160412f����3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
8__inference_batch_normalization_7_layer_call_fn_12160345Y����3�0
)�&
 �
inputs��������� 
p 
� "���������� �
8__inference_batch_normalization_7_layer_call_fn_12160358Y����3�0
)�&
 �
inputs��������� 
p
� "���������� �
K__inference_concatenate_1_layer_call_and_return_conditional_losses_12160479�Z�W
P�M
K�H
"�
inputs/0��������� 
"�
inputs/1��������� 
� "%�"
�
0���������@
� �
0__inference_concatenate_1_layer_call_fn_12160472vZ�W
P�M
K�H
"�
inputs/0��������� 
"�
inputs/1��������� 
� "����������@�
F__inference_dense_10_layer_call_and_return_conditional_losses_12159592\/�,
%�"
 �
inputs���������

� "%�"
�
0��������� 
� ~
+__inference_dense_10_layer_call_fn_12159581O/�,
%�"
 �
inputs���������

� "���������� �
F__inference_dense_11_layer_call_and_return_conditional_losses_12159753\?@/�,
%�"
 �
inputs��������� 
� "%�"
�
0��������� 
� ~
+__inference_dense_11_layer_call_fn_12159742O?@/�,
%�"
 �
inputs��������� 
� "���������� �
F__inference_dense_12_layer_call_and_return_conditional_losses_12160252\st/�,
%�"
 �
inputs��������� 
� "%�"
�
0��������� 
� ~
+__inference_dense_12_layer_call_fn_12160241Ost/�,
%�"
 �
inputs��������� 
� "���������� �
F__inference_dense_13_layer_call_and_return_conditional_losses_12160499^��/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� �
+__inference_dense_13_layer_call_fn_12160488Q��/�,
%�"
 �
inputs���������@
� "�����������
E__inference_dense_7_layer_call_and_return_conditional_losses_12160679f��3�0
)�&
$�!
inputs���������	
� ")�&
�
0���������	
� �
*__inference_dense_7_layer_call_fn_12160648Y��3�0
)�&
$�!
inputs���������	
� "����������	�
E__inference_dense_8_layer_call_and_return_conditional_losses_12160718f��3�0
)�&
$�!
inputs���������	
� ")�&
�
0���������	
� �
*__inference_dense_8_layer_call_fn_12160688Y��3�0
)�&
$�!
inputs���������	
� "����������	�
E__inference_dense_9_layer_call_and_return_conditional_losses_12160232\kl/�,
%�"
 �
inputs���������
� "%�"
�
0��������� 
� }
*__inference_dense_9_layer_call_fn_12160221Okl/�,
%�"
 �
inputs���������
� "���������� �
H__inference_dropout_10_layer_call_and_return_conditional_losses_12160200\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
H__inference_dropout_10_layer_call_and_return_conditional_losses_12160212\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
-__inference_dropout_10_layer_call_fn_12160190O3�0
)�&
 �
inputs��������� 
p 
� "���������� �
-__inference_dropout_10_layer_call_fn_12160195O3�0
)�&
 �
inputs��������� 
p
� "���������� �
H__inference_dropout_11_layer_call_and_return_conditional_losses_12160454\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
H__inference_dropout_11_layer_call_and_return_conditional_losses_12160466\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
-__inference_dropout_11_layer_call_fn_12160444O3�0
)�&
 �
inputs��������� 
p 
� "���������� �
-__inference_dropout_11_layer_call_fn_12160449O3�0
)�&
 �
inputs��������� 
p
� "���������� �
G__inference_dropout_8_layer_call_and_return_conditional_losses_12160427\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
G__inference_dropout_8_layer_call_and_return_conditional_losses_12160439\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� 
,__inference_dropout_8_layer_call_fn_12160417O3�0
)�&
 �
inputs��������� 
p 
� "���������� 
,__inference_dropout_8_layer_call_fn_12160422O3�0
)�&
 �
inputs��������� 
p
� "���������� �
G__inference_dropout_9_layer_call_and_return_conditional_losses_12159687\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
G__inference_dropout_9_layer_call_and_return_conditional_losses_12159699\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� 
,__inference_dropout_9_layer_call_fn_12159677O3�0
)�&
 �
inputs��������� 
p 
� "���������� 
,__inference_dropout_9_layer_call_fn_12159682O3�0
)�&
 �
inputs��������� 
p
� "���������� �
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_12160185{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
=__inference_global_average_pooling1d_1_layer_call_fn_12160179nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
E__inference_model_1_layer_call_and_return_conditional_losses_12158563�D)&('?@��WTVU����������������stkl����|~}��`�]
V�S
I�F
!�
input_3���������	
!�
input_4���������

p 

 
� "%�"
�
0���������
� �
E__inference_model_1_layer_call_and_return_conditional_losses_12158673�D()&'?@��VWTU����������������stkl����~|}��`�]
V�S
I�F
!�
input_3���������	
!�
input_4���������

p

 
� "%�"
�
0���������
� �
E__inference_model_1_layer_call_and_return_conditional_losses_12159123�D)&('?@��WTVU����������������stkl����|~}��b�_
X�U
K�H
"�
inputs/0���������	
"�
inputs/1���������

p 

 
� "%�"
�
0���������
� �
E__inference_model_1_layer_call_and_return_conditional_losses_12159476�D()&'?@��VWTU����������������stkl����~|}��b�_
X�U
K�H
"�
inputs/0���������	
"�
inputs/1���������

p

 
� "%�"
�
0���������
� �
*__inference_model_1_layer_call_fn_12157687�D)&('?@��WTVU����������������stkl����|~}��`�]
V�S
I�F
!�
input_3���������	
!�
input_4���������

p 

 
� "�����������
*__inference_model_1_layer_call_fn_12158453�D()&'?@��VWTU����������������stkl����~|}��`�]
V�S
I�F
!�
input_3���������	
!�
input_4���������

p

 
� "�����������
*__inference_model_1_layer_call_fn_12158773�D)&('?@��WTVU����������������stkl����|~}��b�_
X�U
K�H
"�
inputs/0���������	
"�
inputs/1���������

p 

 
� "�����������
*__inference_model_1_layer_call_fn_12158867�D()&'?@��VWTU����������������stkl����~|}��b�_
X�U
K�H
"�
inputs/0���������	
"�
inputs/1���������

p

 
� "�����������
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156962y����B�?
8�5
+�(
dense_7_input���������	
p 

 
� ")�&
�
0���������	
� �
J__inference_sequential_1_layer_call_and_return_conditional_losses_12156976y����B�?
8�5
+�(
dense_7_input���������	
p

 
� ")�&
�
0���������	
� �
J__inference_sequential_1_layer_call_and_return_conditional_losses_12160582r����;�8
1�.
$�!
inputs���������	
p 

 
� ")�&
�
0���������	
� �
J__inference_sequential_1_layer_call_and_return_conditional_losses_12160639r����;�8
1�.
$�!
inputs���������	
p

 
� ")�&
�
0���������	
� �
/__inference_sequential_1_layer_call_fn_12156875l����B�?
8�5
+�(
dense_7_input���������	
p 

 
� "����������	�
/__inference_sequential_1_layer_call_fn_12156948l����B�?
8�5
+�(
dense_7_input���������	
p

 
� "����������	�
/__inference_sequential_1_layer_call_fn_12160512e����;�8
1�.
$�!
inputs���������	
p 

 
� "����������	�
/__inference_sequential_1_layer_call_fn_12160525e����;�8
1�.
$�!
inputs���������	
p

 
� "����������	�
&__inference_signature_wrapper_12159572�D)&('?@��WTVU����������������stkl����|~}��i�f
� 
_�\
,
input_3!�
input_3���������	
,
input_4!�
input_4���������
"3�0
.
dense_13"�
dense_13����������
\__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12159733]��*�'
 �
�
x���������	
� ")�&
�
0���������	
� �
A__inference_token_and_position_embedding_1_layer_call_fn_12159708P��*�'
 �
�
x���������	
� "����������	�
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12159954� ����������������7�4
-�*
$�!
inputs���������	
p 
� ")�&
�
0���������	
� �
Q__inference_transformer_block_1_layer_call_and_return_conditional_losses_12160094� ����������������7�4
-�*
$�!
inputs���������	
p
� ")�&
�
0���������	
� �
6__inference_transformer_block_1_layer_call_fn_12159790y ����������������7�4
-�*
$�!
inputs���������	
p 
� "����������	�
6__inference_transformer_block_1_layer_call_fn_12159827y ����������������7�4
-�*
$�!
inputs���������	
p
� "����������	