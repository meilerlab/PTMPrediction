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
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 * 
shared_namedense_59/kernel
s
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes

:
 *
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
: *
dtype0
�
batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_33/gamma
�
0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_33/beta
�
/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_33/moving_mean
�
6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_33/moving_variance
�
:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes
: *
dtype0
z
dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_60/kernel
s
#dense_60/kernel/Read/ReadVariableOpReadVariableOpdense_60/kernel*
_output_shapes

:  *
dtype0
r
dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_60/bias
k
!dense_60/bias/Read/ReadVariableOpReadVariableOpdense_60/bias*
_output_shapes
: *
dtype0
�
batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_34/gamma
�
0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_34/beta
�
/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_34/moving_mean
�
6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_34/moving_variance
�
:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes
: *
dtype0
z
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_58/kernel
s
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes

: *
dtype0
r
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
: *
dtype0
z
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_61/kernel
s
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel*
_output_shapes

:  *
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes
: *
dtype0
�
batch_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_32/gamma
�
0batch_normalization_32/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_32/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_32/beta
�
/batch_normalization_32/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_32/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_32/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_32/moving_mean
�
6batch_normalization_32/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_32/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_32/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_32/moving_variance
�
:batch_normalization_32/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_32/moving_variance*
_output_shapes
: *
dtype0
�
batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_35/gamma
�
0batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_35/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_35/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_35/beta
�
/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_35/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_35/moving_mean
�
6batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_35/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_35/moving_variance
�
:batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_35/moving_variance*
_output_shapes
: *
dtype0
z
dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_62/kernel
s
#dense_62/kernel/Read/ReadVariableOpReadVariableOpdense_62/kernel*
_output_shapes

:@*
dtype0
r
dense_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_62/bias
k
!dense_62/bias/Read/ReadVariableOpReadVariableOpdense_62/bias*
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
6token_and_position_embedding_8/embedding_16/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86token_and_position_embedding_8/embedding_16/embeddings
�
Jtoken_and_position_embedding_8/embedding_16/embeddings/Read/ReadVariableOpReadVariableOp6token_and_position_embedding_8/embedding_16/embeddings*
_output_shapes

:*
dtype0
�
6token_and_position_embedding_8/embedding_17/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*G
shared_name86token_and_position_embedding_8/embedding_17/embeddings
�
Jtoken_and_position_embedding_8/embedding_17/embeddings/Read/ReadVariableOpReadVariableOp6token_and_position_embedding_8/embedding_17/embeddings*
_output_shapes

:	*
dtype0
�
*transformer_block_8/attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*transformer_block_8/attention/query/kernel
�
>transformer_block_8/attention/query/kernel/Read/ReadVariableOpReadVariableOp*transformer_block_8/attention/query/kernel*"
_output_shapes
:*
dtype0
�
(transformer_block_8/attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(transformer_block_8/attention/query/bias
�
<transformer_block_8/attention/query/bias/Read/ReadVariableOpReadVariableOp(transformer_block_8/attention/query/bias*
_output_shapes

:*
dtype0
�
(transformer_block_8/attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(transformer_block_8/attention/key/kernel
�
<transformer_block_8/attention/key/kernel/Read/ReadVariableOpReadVariableOp(transformer_block_8/attention/key/kernel*"
_output_shapes
:*
dtype0
�
&transformer_block_8/attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&transformer_block_8/attention/key/bias
�
:transformer_block_8/attention/key/bias/Read/ReadVariableOpReadVariableOp&transformer_block_8/attention/key/bias*
_output_shapes

:*
dtype0
�
*transformer_block_8/attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*transformer_block_8/attention/value/kernel
�
>transformer_block_8/attention/value/kernel/Read/ReadVariableOpReadVariableOp*transformer_block_8/attention/value/kernel*"
_output_shapes
:*
dtype0
�
(transformer_block_8/attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(transformer_block_8/attention/value/bias
�
<transformer_block_8/attention/value/bias/Read/ReadVariableOpReadVariableOp(transformer_block_8/attention/value/bias*
_output_shapes

:*
dtype0
�
5transformer_block_8/attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75transformer_block_8/attention/attention_output/kernel
�
Itransformer_block_8/attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_8/attention/attention_output/kernel*"
_output_shapes
:*
dtype0
�
3transformer_block_8/attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53transformer_block_8/attention/attention_output/bias
�
Gtransformer_block_8/attention/attention_output/bias/Read/ReadVariableOpReadVariableOp3transformer_block_8/attention/attention_output/bias*
_output_shapes
:*
dtype0
z
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_56/kernel
s
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel*
_output_shapes

:*
dtype0
r
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_56/bias
k
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes
:*
dtype0
z
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_57/kernel
s
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes

:*
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
:*
dtype0
�
0transformer_block_8/layer_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20transformer_block_8/layer_normalization_16/gamma
�
Dtransformer_block_8/layer_normalization_16/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_8/layer_normalization_16/gamma*
_output_shapes
:*
dtype0
�
/transformer_block_8/layer_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_block_8/layer_normalization_16/beta
�
Ctransformer_block_8/layer_normalization_16/beta/Read/ReadVariableOpReadVariableOp/transformer_block_8/layer_normalization_16/beta*
_output_shapes
:*
dtype0
�
0transformer_block_8/layer_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20transformer_block_8/layer_normalization_17/gamma
�
Dtransformer_block_8/layer_normalization_17/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_8/layer_normalization_17/gamma*
_output_shapes
:*
dtype0
�
/transformer_block_8/layer_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_block_8/layer_normalization_17/beta
�
Ctransformer_block_8/layer_normalization_17/beta/Read/ReadVariableOpReadVariableOp/transformer_block_8/layer_normalization_17/beta*
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
Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *'
shared_nameAdam/dense_59/kernel/m
�
*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes

:
 *
dtype0
�
Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_33/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_33/gamma/m
�
7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_33/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_33/beta/m
�
6Adam/batch_normalization_33/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/m*
_output_shapes
: *
dtype0
�
Adam/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_60/kernel/m
�
*Adam/dense_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/m*
_output_shapes

:  *
dtype0
�
Adam/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_60/bias/m
y
(Adam/dense_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_34/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_34/gamma/m
�
7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_34/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_34/beta/m
�
6Adam/batch_normalization_34/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/m*
_output_shapes
: *
dtype0
�
Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_58/kernel/m
�
*Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/m*
_output_shapes

: *
dtype0
�
Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_58/bias/m
y
(Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_61/kernel/m
�
*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m*
_output_shapes

:  *
dtype0
�
Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_61/bias/m
y
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_32/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_32/gamma/m
�
7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_32/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_32/beta/m
�
6Adam/batch_normalization_32/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_35/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_35/gamma/m
�
7Adam/batch_normalization_35/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_35/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_35/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_35/beta/m
�
6Adam/batch_normalization_35/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_35/beta/m*
_output_shapes
: *
dtype0
�
Adam/dense_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_62/kernel/m
�
*Adam/dense_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_62/bias/m
y
(Adam/dense_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/m*
_output_shapes
:*
dtype0
�
=Adam/token_and_position_embedding_8/embedding_16/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/token_and_position_embedding_8/embedding_16/embeddings/m
�
QAdam/token_and_position_embedding_8/embedding_16/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_8/embedding_16/embeddings/m*
_output_shapes

:*
dtype0
�
=Adam/token_and_position_embedding_8/embedding_17/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*N
shared_name?=Adam/token_and_position_embedding_8/embedding_17/embeddings/m
�
QAdam/token_and_position_embedding_8/embedding_17/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_8/embedding_17/embeddings/m*
_output_shapes

:	*
dtype0
�
1Adam/transformer_block_8/attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block_8/attention/query/kernel/m
�
EAdam/transformer_block_8/attention/query/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/transformer_block_8/attention/query/kernel/m*"
_output_shapes
:*
dtype0
�
/Adam/transformer_block_8/attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/transformer_block_8/attention/query/bias/m
�
CAdam/transformer_block_8/attention/query/bias/m/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_8/attention/query/bias/m*
_output_shapes

:*
dtype0
�
/Adam/transformer_block_8/attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/transformer_block_8/attention/key/kernel/m
�
CAdam/transformer_block_8/attention/key/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_8/attention/key/kernel/m*"
_output_shapes
:*
dtype0
�
-Adam/transformer_block_8/attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-Adam/transformer_block_8/attention/key/bias/m
�
AAdam/transformer_block_8/attention/key/bias/m/Read/ReadVariableOpReadVariableOp-Adam/transformer_block_8/attention/key/bias/m*
_output_shapes

:*
dtype0
�
1Adam/transformer_block_8/attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block_8/attention/value/kernel/m
�
EAdam/transformer_block_8/attention/value/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/transformer_block_8/attention/value/kernel/m*"
_output_shapes
:*
dtype0
�
/Adam/transformer_block_8/attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/transformer_block_8/attention/value/bias/m
�
CAdam/transformer_block_8/attention/value/bias/m/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_8/attention/value/bias/m*
_output_shapes

:*
dtype0
�
<Adam/transformer_block_8/attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_block_8/attention/attention_output/kernel/m
�
PAdam/transformer_block_8/attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_8/attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
�
:Adam/transformer_block_8/attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_block_8/attention/attention_output/bias/m
�
NAdam/transformer_block_8/attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_8/attention/attention_output/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_56/kernel/m
�
*Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_56/bias/m
y
(Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_57/kernel/m
�
*Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_57/bias/m
y
(Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/m*
_output_shapes
:*
dtype0
�
7Adam/transformer_block_8/layer_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_block_8/layer_normalization_16/gamma/m
�
KAdam/transformer_block_8/layer_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_16/gamma/m*
_output_shapes
:*
dtype0
�
6Adam/transformer_block_8/layer_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_8/layer_normalization_16/beta/m
�
JAdam/transformer_block_8/layer_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_16/beta/m*
_output_shapes
:*
dtype0
�
7Adam/transformer_block_8/layer_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_block_8/layer_normalization_17/gamma/m
�
KAdam/transformer_block_8/layer_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_17/gamma/m*
_output_shapes
:*
dtype0
�
6Adam/transformer_block_8/layer_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_8/layer_normalization_17/beta/m
�
JAdam/transformer_block_8/layer_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_17/beta/m*
_output_shapes
:*
dtype0
�
Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *'
shared_nameAdam/dense_59/kernel/v
�
*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes

:
 *
dtype0
�
Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_33/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_33/gamma/v
�
7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_33/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_33/beta/v
�
6Adam/batch_normalization_33/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/v*
_output_shapes
: *
dtype0
�
Adam/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_60/kernel/v
�
*Adam/dense_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/v*
_output_shapes

:  *
dtype0
�
Adam/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_60/bias/v
y
(Adam/dense_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_34/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_34/gamma/v
�
7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_34/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_34/beta/v
�
6Adam/batch_normalization_34/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/v*
_output_shapes
: *
dtype0
�
Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_58/kernel/v
�
*Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/v*
_output_shapes

: *
dtype0
�
Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_58/bias/v
y
(Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_61/kernel/v
�
*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v*
_output_shapes

:  *
dtype0
�
Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_61/bias/v
y
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_32/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_32/gamma/v
�
7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_32/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_32/beta/v
�
6Adam/batch_normalization_32/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_35/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_35/gamma/v
�
7Adam/batch_normalization_35/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_35/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_35/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_35/beta/v
�
6Adam/batch_normalization_35/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_35/beta/v*
_output_shapes
: *
dtype0
�
Adam/dense_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_62/kernel/v
�
*Adam/dense_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_62/bias/v
y
(Adam/dense_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/v*
_output_shapes
:*
dtype0
�
=Adam/token_and_position_embedding_8/embedding_16/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/token_and_position_embedding_8/embedding_16/embeddings/v
�
QAdam/token_and_position_embedding_8/embedding_16/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_8/embedding_16/embeddings/v*
_output_shapes

:*
dtype0
�
=Adam/token_and_position_embedding_8/embedding_17/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*N
shared_name?=Adam/token_and_position_embedding_8/embedding_17/embeddings/v
�
QAdam/token_and_position_embedding_8/embedding_17/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_8/embedding_17/embeddings/v*
_output_shapes

:	*
dtype0
�
1Adam/transformer_block_8/attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block_8/attention/query/kernel/v
�
EAdam/transformer_block_8/attention/query/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/transformer_block_8/attention/query/kernel/v*"
_output_shapes
:*
dtype0
�
/Adam/transformer_block_8/attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/transformer_block_8/attention/query/bias/v
�
CAdam/transformer_block_8/attention/query/bias/v/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_8/attention/query/bias/v*
_output_shapes

:*
dtype0
�
/Adam/transformer_block_8/attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/transformer_block_8/attention/key/kernel/v
�
CAdam/transformer_block_8/attention/key/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_8/attention/key/kernel/v*"
_output_shapes
:*
dtype0
�
-Adam/transformer_block_8/attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-Adam/transformer_block_8/attention/key/bias/v
�
AAdam/transformer_block_8/attention/key/bias/v/Read/ReadVariableOpReadVariableOp-Adam/transformer_block_8/attention/key/bias/v*
_output_shapes

:*
dtype0
�
1Adam/transformer_block_8/attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block_8/attention/value/kernel/v
�
EAdam/transformer_block_8/attention/value/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/transformer_block_8/attention/value/kernel/v*"
_output_shapes
:*
dtype0
�
/Adam/transformer_block_8/attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/transformer_block_8/attention/value/bias/v
�
CAdam/transformer_block_8/attention/value/bias/v/Read/ReadVariableOpReadVariableOp/Adam/transformer_block_8/attention/value/bias/v*
_output_shapes

:*
dtype0
�
<Adam/transformer_block_8/attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_block_8/attention/attention_output/kernel/v
�
PAdam/transformer_block_8/attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_8/attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
�
:Adam/transformer_block_8/attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_block_8/attention/attention_output/bias/v
�
NAdam/transformer_block_8/attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_8/attention/attention_output/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_56/kernel/v
�
*Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_56/bias/v
y
(Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_57/kernel/v
�
*Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_57/bias/v
y
(Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/v*
_output_shapes
:*
dtype0
�
7Adam/transformer_block_8/layer_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_block_8/layer_normalization_16/gamma/v
�
KAdam/transformer_block_8/layer_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_16/gamma/v*
_output_shapes
:*
dtype0
�
6Adam/transformer_block_8/layer_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_8/layer_normalization_16/beta/v
�
JAdam/transformer_block_8/layer_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_16/beta/v*
_output_shapes
:*
dtype0
�
7Adam/transformer_block_8/layer_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_block_8/layer_normalization_17/gamma/v
�
KAdam/transformer_block_8/layer_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_17/gamma/v*
_output_shapes
:*
dtype0
�
6Adam/transformer_block_8/layer_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_8/layer_normalization_17/beta/v
�
JAdam/transformer_block_8/layer_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_17/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B�
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
VARIABLE_VALUEdense_59/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_59/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ke
VARIABLE_VALUEbatch_normalization_33/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_33/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_33/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_33/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_60/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_60/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ke
VARIABLE_VALUEbatch_normalization_34/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_34/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_34/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_34/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
_Y
VARIABLE_VALUEdense_58/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_58/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_61/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_61/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ke
VARIABLE_VALUEbatch_normalization_32/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_32/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_32/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_32/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
ke
VARIABLE_VALUEbatch_normalization_35/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_35/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_35/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_35/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_62/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_62/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
vp
VARIABLE_VALUE6token_and_position_embedding_8/embedding_16/embeddings&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6token_and_position_embedding_8/embedding_17/embeddings&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*transformer_block_8/attention/query/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(transformer_block_8/attention/query/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(transformer_block_8/attention/key/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&transformer_block_8/attention/key/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*transformer_block_8/attention/value/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(transformer_block_8/attention/value/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5transformer_block_8/attention/attention_output/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3transformer_block_8/attention/attention_output/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_56/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_56/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_57/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_57/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_block_8/layer_normalization_16/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_block_8/layer_normalization_16/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_block_8/layer_normalization_17/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_block_8/layer_normalization_17/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_59/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_59/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_33/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_60/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_60/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_34/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_34/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_58/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_58/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_61/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_61/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_32/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_35/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_35/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_62/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_62/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/token_and_position_embedding_8/embedding_16/embeddings/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/token_and_position_embedding_8/embedding_17/embeddings/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/transformer_block_8/attention/query/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_8/attention/query/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_8/attention/key/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/transformer_block_8/attention/key/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/transformer_block_8/attention/value/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_8/attention/value/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/transformer_block_8/attention/attention_output/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/transformer_block_8/attention/attention_output/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_56/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_56/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_57/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_57/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_16/gamma/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_16/beta/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_17/gamma/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_17/beta/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_59/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_59/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_33/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_60/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_60/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_34/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_34/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_58/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_58/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_61/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_61/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_32/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_35/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_35/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_62/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_62/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/token_and_position_embedding_8/embedding_16/embeddings/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/token_and_position_embedding_8/embedding_17/embeddings/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/transformer_block_8/attention/query/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_8/attention/query/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_8/attention/key/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/transformer_block_8/attention/key/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/transformer_block_8/attention/value/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/Adam/transformer_block_8/attention/value/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/transformer_block_8/attention/attention_output/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/transformer_block_8/attention/attention_output/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_56/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_56/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_57/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_57/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_16/gamma/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_16/beta/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_17/gamma/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_17/beta/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
serving_default_input_17Placeholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
{
serving_default_input_18Placeholder*'
_output_shapes
:���������
*
dtype0*
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17serving_default_input_18dense_59/kerneldense_59/bias&batch_normalization_33/moving_variancebatch_normalization_33/gamma"batch_normalization_33/moving_meanbatch_normalization_33/betadense_60/kerneldense_60/bias6token_and_position_embedding_8/embedding_17/embeddings6token_and_position_embedding_8/embedding_16/embeddings&batch_normalization_34/moving_variancebatch_normalization_34/gamma"batch_normalization_34/moving_meanbatch_normalization_34/beta*transformer_block_8/attention/query/kernel(transformer_block_8/attention/query/bias(transformer_block_8/attention/key/kernel&transformer_block_8/attention/key/bias*transformer_block_8/attention/value/kernel(transformer_block_8/attention/value/bias5transformer_block_8/attention/attention_output/kernel3transformer_block_8/attention/attention_output/bias0transformer_block_8/layer_normalization_16/gamma/transformer_block_8/layer_normalization_16/betadense_56/kerneldense_56/biasdense_57/kerneldense_57/bias0transformer_block_8/layer_normalization_17/gamma/transformer_block_8/layer_normalization_17/betadense_61/kerneldense_61/biasdense_58/kerneldense_58/bias&batch_normalization_35/moving_variancebatch_normalization_35/gamma"batch_normalization_35/moving_meanbatch_normalization_35/beta&batch_normalization_32/moving_variancebatch_normalization_32/gamma"batch_normalization_32/moving_meanbatch_normalization_32/betadense_62/kerneldense_62/bias*9
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
GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_9510606
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�7
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOp0batch_normalization_33/gamma/Read/ReadVariableOp/batch_normalization_33/beta/Read/ReadVariableOp6batch_normalization_33/moving_mean/Read/ReadVariableOp:batch_normalization_33/moving_variance/Read/ReadVariableOp#dense_60/kernel/Read/ReadVariableOp!dense_60/bias/Read/ReadVariableOp0batch_normalization_34/gamma/Read/ReadVariableOp/batch_normalization_34/beta/Read/ReadVariableOp6batch_normalization_34/moving_mean/Read/ReadVariableOp:batch_normalization_34/moving_variance/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOp#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOp0batch_normalization_32/gamma/Read/ReadVariableOp/batch_normalization_32/beta/Read/ReadVariableOp6batch_normalization_32/moving_mean/Read/ReadVariableOp:batch_normalization_32/moving_variance/Read/ReadVariableOp0batch_normalization_35/gamma/Read/ReadVariableOp/batch_normalization_35/beta/Read/ReadVariableOp6batch_normalization_35/moving_mean/Read/ReadVariableOp:batch_normalization_35/moving_variance/Read/ReadVariableOp#dense_62/kernel/Read/ReadVariableOp!dense_62/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpJtoken_and_position_embedding_8/embedding_16/embeddings/Read/ReadVariableOpJtoken_and_position_embedding_8/embedding_17/embeddings/Read/ReadVariableOp>transformer_block_8/attention/query/kernel/Read/ReadVariableOp<transformer_block_8/attention/query/bias/Read/ReadVariableOp<transformer_block_8/attention/key/kernel/Read/ReadVariableOp:transformer_block_8/attention/key/bias/Read/ReadVariableOp>transformer_block_8/attention/value/kernel/Read/ReadVariableOp<transformer_block_8/attention/value/bias/Read/ReadVariableOpItransformer_block_8/attention/attention_output/kernel/Read/ReadVariableOpGtransformer_block_8/attention/attention_output/bias/Read/ReadVariableOp#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOpDtransformer_block_8/layer_normalization_16/gamma/Read/ReadVariableOpCtransformer_block_8/layer_normalization_16/beta/Read/ReadVariableOpDtransformer_block_8/layer_normalization_17/gamma/Read/ReadVariableOpCtransformer_block_8/layer_normalization_17/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_33/beta/m/Read/ReadVariableOp*Adam/dense_60/kernel/m/Read/ReadVariableOp(Adam/dense_60/bias/m/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_34/beta/m/Read/ReadVariableOp*Adam/dense_58/kernel/m/Read/ReadVariableOp(Adam/dense_58/bias/m/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_32/beta/m/Read/ReadVariableOp7Adam/batch_normalization_35/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_35/beta/m/Read/ReadVariableOp*Adam/dense_62/kernel/m/Read/ReadVariableOp(Adam/dense_62/bias/m/Read/ReadVariableOpQAdam/token_and_position_embedding_8/embedding_16/embeddings/m/Read/ReadVariableOpQAdam/token_and_position_embedding_8/embedding_17/embeddings/m/Read/ReadVariableOpEAdam/transformer_block_8/attention/query/kernel/m/Read/ReadVariableOpCAdam/transformer_block_8/attention/query/bias/m/Read/ReadVariableOpCAdam/transformer_block_8/attention/key/kernel/m/Read/ReadVariableOpAAdam/transformer_block_8/attention/key/bias/m/Read/ReadVariableOpEAdam/transformer_block_8/attention/value/kernel/m/Read/ReadVariableOpCAdam/transformer_block_8/attention/value/bias/m/Read/ReadVariableOpPAdam/transformer_block_8/attention/attention_output/kernel/m/Read/ReadVariableOpNAdam/transformer_block_8/attention/attention_output/bias/m/Read/ReadVariableOp*Adam/dense_56/kernel/m/Read/ReadVariableOp(Adam/dense_56/bias/m/Read/ReadVariableOp*Adam/dense_57/kernel/m/Read/ReadVariableOp(Adam/dense_57/bias/m/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_16/gamma/m/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_16/beta/m/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_17/gamma/m/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_17/beta/m/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_33/beta/v/Read/ReadVariableOp*Adam/dense_60/kernel/v/Read/ReadVariableOp(Adam/dense_60/bias/v/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_34/beta/v/Read/ReadVariableOp*Adam/dense_58/kernel/v/Read/ReadVariableOp(Adam/dense_58/bias/v/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_32/beta/v/Read/ReadVariableOp7Adam/batch_normalization_35/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_35/beta/v/Read/ReadVariableOp*Adam/dense_62/kernel/v/Read/ReadVariableOp(Adam/dense_62/bias/v/Read/ReadVariableOpQAdam/token_and_position_embedding_8/embedding_16/embeddings/v/Read/ReadVariableOpQAdam/token_and_position_embedding_8/embedding_17/embeddings/v/Read/ReadVariableOpEAdam/transformer_block_8/attention/query/kernel/v/Read/ReadVariableOpCAdam/transformer_block_8/attention/query/bias/v/Read/ReadVariableOpCAdam/transformer_block_8/attention/key/kernel/v/Read/ReadVariableOpAAdam/transformer_block_8/attention/key/bias/v/Read/ReadVariableOpEAdam/transformer_block_8/attention/value/kernel/v/Read/ReadVariableOpCAdam/transformer_block_8/attention/value/bias/v/Read/ReadVariableOpPAdam/transformer_block_8/attention/attention_output/kernel/v/Read/ReadVariableOpNAdam/transformer_block_8/attention/attention_output/bias/v/Read/ReadVariableOp*Adam/dense_56/kernel/v/Read/ReadVariableOp(Adam/dense_56/bias/v/Read/ReadVariableOp*Adam/dense_57/kernel/v/Read/ReadVariableOp(Adam/dense_57/bias/v/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_16/gamma/v/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_16/beta/v/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_17/gamma/v/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_17/beta/v/Read/ReadVariableOpConst*�
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_9512145
�$
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_59/kerneldense_59/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_variancedense_60/kerneldense_60/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_variancedense_58/kerneldense_58/biasdense_61/kerneldense_61/biasbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_variancebatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_62/kerneldense_62/biasbeta_1beta_2decaylearning_rate	Adam/iter6token_and_position_embedding_8/embedding_16/embeddings6token_and_position_embedding_8/embedding_17/embeddings*transformer_block_8/attention/query/kernel(transformer_block_8/attention/query/bias(transformer_block_8/attention/key/kernel&transformer_block_8/attention/key/bias*transformer_block_8/attention/value/kernel(transformer_block_8/attention/value/bias5transformer_block_8/attention/attention_output/kernel3transformer_block_8/attention/attention_output/biasdense_56/kerneldense_56/biasdense_57/kerneldense_57/bias0transformer_block_8/layer_normalization_16/gamma/transformer_block_8/layer_normalization_16/beta0transformer_block_8/layer_normalization_17/gamma/transformer_block_8/layer_normalization_17/betatotalcountAdam/dense_59/kernel/mAdam/dense_59/bias/m#Adam/batch_normalization_33/gamma/m"Adam/batch_normalization_33/beta/mAdam/dense_60/kernel/mAdam/dense_60/bias/m#Adam/batch_normalization_34/gamma/m"Adam/batch_normalization_34/beta/mAdam/dense_58/kernel/mAdam/dense_58/bias/mAdam/dense_61/kernel/mAdam/dense_61/bias/m#Adam/batch_normalization_32/gamma/m"Adam/batch_normalization_32/beta/m#Adam/batch_normalization_35/gamma/m"Adam/batch_normalization_35/beta/mAdam/dense_62/kernel/mAdam/dense_62/bias/m=Adam/token_and_position_embedding_8/embedding_16/embeddings/m=Adam/token_and_position_embedding_8/embedding_17/embeddings/m1Adam/transformer_block_8/attention/query/kernel/m/Adam/transformer_block_8/attention/query/bias/m/Adam/transformer_block_8/attention/key/kernel/m-Adam/transformer_block_8/attention/key/bias/m1Adam/transformer_block_8/attention/value/kernel/m/Adam/transformer_block_8/attention/value/bias/m<Adam/transformer_block_8/attention/attention_output/kernel/m:Adam/transformer_block_8/attention/attention_output/bias/mAdam/dense_56/kernel/mAdam/dense_56/bias/mAdam/dense_57/kernel/mAdam/dense_57/bias/m7Adam/transformer_block_8/layer_normalization_16/gamma/m6Adam/transformer_block_8/layer_normalization_16/beta/m7Adam/transformer_block_8/layer_normalization_17/gamma/m6Adam/transformer_block_8/layer_normalization_17/beta/mAdam/dense_59/kernel/vAdam/dense_59/bias/v#Adam/batch_normalization_33/gamma/v"Adam/batch_normalization_33/beta/vAdam/dense_60/kernel/vAdam/dense_60/bias/v#Adam/batch_normalization_34/gamma/v"Adam/batch_normalization_34/beta/vAdam/dense_58/kernel/vAdam/dense_58/bias/vAdam/dense_61/kernel/vAdam/dense_61/bias/v#Adam/batch_normalization_32/gamma/v"Adam/batch_normalization_32/beta/v#Adam/batch_normalization_35/gamma/v"Adam/batch_normalization_35/beta/vAdam/dense_62/kernel/vAdam/dense_62/bias/v=Adam/token_and_position_embedding_8/embedding_16/embeddings/v=Adam/token_and_position_embedding_8/embedding_17/embeddings/v1Adam/transformer_block_8/attention/query/kernel/v/Adam/transformer_block_8/attention/query/bias/v/Adam/transformer_block_8/attention/key/kernel/v-Adam/transformer_block_8/attention/key/bias/v1Adam/transformer_block_8/attention/value/kernel/v/Adam/transformer_block_8/attention/value/bias/v<Adam/transformer_block_8/attention/attention_output/kernel/v:Adam/transformer_block_8/attention/attention_output/bias/vAdam/dense_56/kernel/vAdam/dense_56/bias/vAdam/dense_57/kernel/vAdam/dense_57/bias/v7Adam/transformer_block_8/layer_normalization_16/gamma/v6Adam/transformer_block_8/layer_normalization_16/beta/v7Adam/transformer_block_8/layer_normalization_17/gamma/v6Adam/transformer_block_8/layer_normalization_17/beta/v*�
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_9512524��$
�
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_9508010
dense_56_input"
dense_56_9507999:
dense_56_9508001:"
dense_57_9508004:
dense_57_9508006:
identity�� dense_56/StatefulPartitionedCall� dense_57/StatefulPartitionedCall�
 dense_56/StatefulPartitionedCallStatefulPartitionedCalldense_56_inputdense_56_9507999dense_56_9508001*
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
E__inference_dense_56_layer_call_and_return_conditional_losses_9507855�
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_9508004dense_57_9508006*
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
E__inference_dense_57_layer_call_and_return_conditional_losses_9507891|
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:[ W
+
_output_shapes
:���������	
(
_user_specified_namedense_56_input
�	
f
G__inference_dropout_53_layer_call_and_return_conditional_losses_9508758

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
E__inference_dense_56_layer_call_and_return_conditional_losses_9511713

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
�`
�
D__inference_model_8_layer_call_and_return_conditional_losses_9508630

inputs
inputs_1"
dense_59_9508290:
 
dense_59_9508292: ,
batch_normalization_33_9508295: ,
batch_normalization_33_9508297: ,
batch_normalization_33_9508299: ,
batch_normalization_33_9508301: "
dense_60_9508323:  
dense_60_9508325: 8
&token_and_position_embedding_8_9508354:	8
&token_and_position_embedding_8_9508356:,
batch_normalization_34_9508359: ,
batch_normalization_34_9508361: ,
batch_normalization_34_9508363: ,
batch_normalization_34_9508365: 1
transformer_block_8_9508496:-
transformer_block_8_9508498:1
transformer_block_8_9508500:-
transformer_block_8_9508502:1
transformer_block_8_9508504:-
transformer_block_8_9508506:1
transformer_block_8_9508508:)
transformer_block_8_9508510:)
transformer_block_8_9508512:)
transformer_block_8_9508514:-
transformer_block_8_9508516:)
transformer_block_8_9508518:-
transformer_block_8_9508520:)
transformer_block_8_9508522:)
transformer_block_8_9508524:)
transformer_block_8_9508526:"
dense_61_9508549:  
dense_61_9508551: "
dense_58_9508566: 
dense_58_9508568: ,
batch_normalization_35_9508571: ,
batch_normalization_35_9508573: ,
batch_normalization_35_9508575: ,
batch_normalization_35_9508577: ,
batch_normalization_32_9508580: ,
batch_normalization_32_9508582: ,
batch_normalization_32_9508584: ,
batch_normalization_32_9508586: "
dense_62_9508624:@
dense_62_9508626:
identity��.batch_normalization_32/StatefulPartitionedCall�.batch_normalization_33/StatefulPartitionedCall�.batch_normalization_34/StatefulPartitionedCall�.batch_normalization_35/StatefulPartitionedCall� dense_58/StatefulPartitionedCall� dense_59/StatefulPartitionedCall� dense_60/StatefulPartitionedCall� dense_61/StatefulPartitionedCall� dense_62/StatefulPartitionedCall�6token_and_position_embedding_8/StatefulPartitionedCall�+transformer_block_8/StatefulPartitionedCall�
 dense_59/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_59_9508290dense_59_9508292*
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
E__inference_dense_59_layer_call_and_return_conditional_losses_9508289�
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0batch_normalization_33_9508295batch_normalization_33_9508297batch_normalization_33_9508299batch_normalization_33_9508301*
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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9507759�
dropout_51/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_9508309�
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#dropout_51/PartitionedCall:output:0dense_60_9508323dense_60_9508325*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_9508322�
6token_and_position_embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputs&token_and_position_embedding_8_9508354&token_and_position_embedding_8_9508356*
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
GPU 2J 8� *d
f_R]
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9508353�
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_34_9508359batch_normalization_34_9508361batch_normalization_34_9508363batch_normalization_34_9508365*
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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9508034�
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_8/StatefulPartitionedCall:output:0transformer_block_8_9508496transformer_block_8_9508498transformer_block_8_9508500transformer_block_8_9508502transformer_block_8_9508504transformer_block_8_9508506transformer_block_8_9508508transformer_block_8_9508510transformer_block_8_9508512transformer_block_8_9508514transformer_block_8_9508516transformer_block_8_9508518transformer_block_8_9508520transformer_block_8_9508522transformer_block_8_9508524transformer_block_8_9508526*
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
GPU 2J 8� *Y
fTRR
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9508495�
dropout_52/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
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
G__inference_dropout_52_layer_call_and_return_conditional_losses_9508534�
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *`
f[RY
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9508102�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall#dropout_52/PartitionedCall:output:0dense_61_9508549dense_61_9508551*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_9508548�
 dense_58/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_58_9508566dense_58_9508568*
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
E__inference_dense_58_layer_call_and_return_conditional_losses_9508565�
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_35_9508571batch_normalization_35_9508573batch_normalization_35_9508575batch_normalization_35_9508577*
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9508211�
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_32_9508580batch_normalization_32_9508582batch_normalization_32_9508584batch_normalization_32_9508586*
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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9508129�
dropout_50/PartitionedCallPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_9508594�
dropout_53/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
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
G__inference_dropout_53_layer_call_and_return_conditional_losses_9508601�
concatenate_8/PartitionedCallPartitionedCall#dropout_50/PartitionedCall:output:0#dropout_53/PartitionedCall:output:0*
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9508610�
 dense_62/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_62_9508624dense_62_9508626*
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
GPU 2J 8� *N
fIRG
E__inference_dense_62_layer_call_and_return_conditional_losses_9508623x
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall7^token_and_position_embedding_8/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2p
6token_and_position_embedding_8/StatefulPartitionedCall6token_and_position_embedding_8/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
H
,__inference_dropout_51_layer_call_fn_9510711

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
G__inference_dropout_51_layer_call_and_return_conditional_losses_9508309`
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
�
�
8__inference_batch_normalization_35_layer_call_fn_9511392

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9508258o
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
8__inference_batch_normalization_32_layer_call_fn_9511312

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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9508176o
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
.__inference_sequential_8_layer_call_fn_9511546

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
GPU 2J 8� *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507898s
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
�	
f
G__inference_dropout_52_layer_call_and_return_conditional_losses_9508824

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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9510767
x7
%embedding_17_embedding_lookup_9510754:	7
%embedding_16_embedding_lookup_9510760:
identity��embedding_16/embedding_lookup�embedding_17/embedding_lookup6
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
embedding_17/embedding_lookupResourceGather%embedding_17_embedding_lookup_9510754range:output:0*
Tindices0*8
_class.
,*loc:@embedding_17/embedding_lookup/9510754*
_output_shapes

:	*
dtype0�
&embedding_17/embedding_lookup/IdentityIdentity&embedding_17/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_17/embedding_lookup/9510754*
_output_shapes

:	�
(embedding_17/embedding_lookup/Identity_1Identity/embedding_17/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	]
embedding_16/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:���������	�
embedding_16/embedding_lookupResourceGather%embedding_16_embedding_lookup_9510760embedding_16/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_16/embedding_lookup/9510760*+
_output_shapes
:���������	*
dtype0�
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_16/embedding_lookup/9510760*+
_output_shapes
:���������	�
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
addAddV21embedding_16/embedding_lookup/Identity_1:output:01embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp^embedding_16/embedding_lookup^embedding_17/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2>
embedding_17/embedding_lookupembedding_17/embedding_lookup:J F
'
_output_shapes
:���������	

_user_specified_namex
�	
f
G__inference_dropout_53_layer_call_and_return_conditional_losses_9511500

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
�
X
<__inference_global_average_pooling1d_8_layer_call_fn_9511213

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
GPU 2J 8� *`
f[RY
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9508102i
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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9508176

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
,__inference_dropout_52_layer_call_fn_9511229

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
G__inference_dropout_52_layer_call_and_return_conditional_losses_9508824o
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
.__inference_sequential_8_layer_call_fn_9507909
dense_56_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_56_inputunknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8� *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507898s
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
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������	
(
_user_specified_namedense_56_input
�
�
8__inference_batch_normalization_32_layer_call_fn_9511299

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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9508129o
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
e
G__inference_dropout_51_layer_call_and_return_conditional_losses_9508309

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
�
�
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9507759

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
G__inference_dropout_52_layer_call_and_return_conditional_losses_9511246

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
�
e
,__inference_dropout_50_layer_call_fn_9511456

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
G__inference_dropout_50_layer_call_and_return_conditional_losses_9508781o
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
Ƿ
�1
D__inference_model_8_layer_call_and_return_conditional_losses_9510510
inputs_0
inputs_19
'dense_59_matmul_readvariableop_resource:
 6
(dense_59_biasadd_readvariableop_resource: L
>batch_normalization_33_assignmovingavg_readvariableop_resource: N
@batch_normalization_33_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_33_batchnorm_mul_readvariableop_resource: F
8batch_normalization_33_batchnorm_readvariableop_resource: 9
'dense_60_matmul_readvariableop_resource:  6
(dense_60_biasadd_readvariableop_resource: V
Dtoken_and_position_embedding_8_embedding_17_embedding_lookup_9510222:	V
Dtoken_and_position_embedding_8_embedding_16_embedding_lookup_9510228:L
>batch_normalization_34_assignmovingavg_readvariableop_resource: N
@batch_normalization_34_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_34_batchnorm_mul_readvariableop_resource: F
8batch_normalization_34_batchnorm_readvariableop_resource: _
Itransformer_block_8_attention_query_einsum_einsum_readvariableop_resource:Q
?transformer_block_8_attention_query_add_readvariableop_resource:]
Gtransformer_block_8_attention_key_einsum_einsum_readvariableop_resource:O
=transformer_block_8_attention_key_add_readvariableop_resource:_
Itransformer_block_8_attention_value_einsum_einsum_readvariableop_resource:Q
?transformer_block_8_attention_value_add_readvariableop_resource:j
Ttransformer_block_8_attention_attention_output_einsum_einsum_readvariableop_resource:X
Jtransformer_block_8_attention_attention_output_add_readvariableop_resource:^
Ptransformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource:Z
Ltransformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource:]
Ktransformer_block_8_sequential_8_dense_56_tensordot_readvariableop_resource:W
Itransformer_block_8_sequential_8_dense_56_biasadd_readvariableop_resource:]
Ktransformer_block_8_sequential_8_dense_57_tensordot_readvariableop_resource:W
Itransformer_block_8_sequential_8_dense_57_biasadd_readvariableop_resource:^
Ptransformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource:Z
Ltransformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource:9
'dense_61_matmul_readvariableop_resource:  6
(dense_61_biasadd_readvariableop_resource: 9
'dense_58_matmul_readvariableop_resource: 6
(dense_58_biasadd_readvariableop_resource: L
>batch_normalization_35_assignmovingavg_readvariableop_resource: N
@batch_normalization_35_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_35_batchnorm_mul_readvariableop_resource: F
8batch_normalization_35_batchnorm_readvariableop_resource: L
>batch_normalization_32_assignmovingavg_readvariableop_resource: N
@batch_normalization_32_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_32_batchnorm_mul_readvariableop_resource: F
8batch_normalization_32_batchnorm_readvariableop_resource: 9
'dense_62_matmul_readvariableop_resource:@6
(dense_62_biasadd_readvariableop_resource:
identity��&batch_normalization_32/AssignMovingAvg�5batch_normalization_32/AssignMovingAvg/ReadVariableOp�(batch_normalization_32/AssignMovingAvg_1�7batch_normalization_32/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_32/batchnorm/ReadVariableOp�3batch_normalization_32/batchnorm/mul/ReadVariableOp�&batch_normalization_33/AssignMovingAvg�5batch_normalization_33/AssignMovingAvg/ReadVariableOp�(batch_normalization_33/AssignMovingAvg_1�7batch_normalization_33/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_33/batchnorm/ReadVariableOp�3batch_normalization_33/batchnorm/mul/ReadVariableOp�&batch_normalization_34/AssignMovingAvg�5batch_normalization_34/AssignMovingAvg/ReadVariableOp�(batch_normalization_34/AssignMovingAvg_1�7batch_normalization_34/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_34/batchnorm/ReadVariableOp�3batch_normalization_34/batchnorm/mul/ReadVariableOp�&batch_normalization_35/AssignMovingAvg�5batch_normalization_35/AssignMovingAvg/ReadVariableOp�(batch_normalization_35/AssignMovingAvg_1�7batch_normalization_35/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_35/batchnorm/ReadVariableOp�3batch_normalization_35/batchnorm/mul/ReadVariableOp�dense_58/BiasAdd/ReadVariableOp�dense_58/MatMul/ReadVariableOp�dense_59/BiasAdd/ReadVariableOp�dense_59/MatMul/ReadVariableOp�dense_60/BiasAdd/ReadVariableOp�dense_60/MatMul/ReadVariableOp�dense_61/BiasAdd/ReadVariableOp�dense_61/MatMul/ReadVariableOp�dense_62/BiasAdd/ReadVariableOp�dense_62/MatMul/ReadVariableOp�<token_and_position_embedding_8/embedding_16/embedding_lookup�<token_and_position_embedding_8/embedding_17/embedding_lookup�Atransformer_block_8/attention/attention_output/add/ReadVariableOp�Ktransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp�4transformer_block_8/attention/key/add/ReadVariableOp�>transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp�6transformer_block_8/attention/query/add/ReadVariableOp�@transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp�6transformer_block_8/attention/value/add/ReadVariableOp�@transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp�Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp�Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp�Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp�Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp�@transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp�Btransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp�@transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp�Btransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp�
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype0}
dense_59/MatMulMatMulinputs_1&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_59/ReluReludense_59/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 
5batch_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_33/moments/meanMeandense_59/Relu:activations:0>batch_normalization_33/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
+batch_normalization_33/moments/StopGradientStopGradient,batch_normalization_33/moments/mean:output:0*
T0*
_output_shapes

: �
0batch_normalization_33/moments/SquaredDifferenceSquaredDifferencedense_59/Relu:activations:04batch_normalization_33/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
9batch_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_33/moments/varianceMean4batch_normalization_33/moments/SquaredDifference:z:0Bbatch_normalization_33/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
&batch_normalization_33/moments/SqueezeSqueeze,batch_normalization_33/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
(batch_normalization_33/moments/Squeeze_1Squeeze0batch_normalization_33/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 q
,batch_normalization_33/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_33/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_33_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
*batch_normalization_33/AssignMovingAvg/subSub=batch_normalization_33/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_33/moments/Squeeze:output:0*
T0*
_output_shapes
: �
*batch_normalization_33/AssignMovingAvg/mulMul.batch_normalization_33/AssignMovingAvg/sub:z:05batch_normalization_33/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
&batch_normalization_33/AssignMovingAvgAssignSubVariableOp>batch_normalization_33_assignmovingavg_readvariableop_resource.batch_normalization_33/AssignMovingAvg/mul:z:06^batch_normalization_33/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_33/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_33/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_33_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
,batch_normalization_33/AssignMovingAvg_1/subSub?batch_normalization_33/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_33/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
,batch_normalization_33/AssignMovingAvg_1/mulMul0batch_normalization_33/AssignMovingAvg_1/sub:z:07batch_normalization_33/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
(batch_normalization_33/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_33_assignmovingavg_1_readvariableop_resource0batch_normalization_33/AssignMovingAvg_1/mul:z:08^batch_normalization_33/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_33/batchnorm/addAddV21batch_normalization_33/moments/Squeeze_1:output:0/batch_normalization_33/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_33/batchnorm/RsqrtRsqrt(batch_normalization_33/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_33/batchnorm/mulMul*batch_normalization_33/batchnorm/Rsqrt:y:0;batch_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_33/batchnorm/mul_1Muldense_59/Relu:activations:0(batch_normalization_33/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
&batch_normalization_33/batchnorm/mul_2Mul/batch_normalization_33/moments/Squeeze:output:0(batch_normalization_33/batchnorm/mul:z:0*
T0*
_output_shapes
: �
/batch_normalization_33/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_33/batchnorm/subSub7batch_normalization_33/batchnorm/ReadVariableOp:value:0*batch_normalization_33/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_33/batchnorm/add_1AddV2*batch_normalization_33/batchnorm/mul_1:z:0(batch_normalization_33/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� ]
dropout_51/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_51/dropout/MulMul*batch_normalization_33/batchnorm/add_1:z:0!dropout_51/dropout/Const:output:0*
T0*'
_output_shapes
:��������� r
dropout_51/dropout/ShapeShape*batch_normalization_33/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_51/dropout/random_uniform/RandomUniformRandomUniform!dropout_51/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0f
!dropout_51/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_51/dropout/GreaterEqualGreaterEqual8dropout_51/dropout/random_uniform/RandomUniform:output:0*dropout_51/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
dropout_51/dropout/CastCast#dropout_51/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
dropout_51/dropout/Mul_1Muldropout_51/dropout/Mul:z:0dropout_51/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� �
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_60/MatMulMatMuldropout_51/dropout/Mul_1:z:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:��������� \
$token_and_position_embedding_8/ShapeShapeinputs_0*
T0*
_output_shapes
:�
2token_and_position_embedding_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������~
4token_and_position_embedding_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4token_and_position_embedding_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,token_and_position_embedding_8/strided_sliceStridedSlice-token_and_position_embedding_8/Shape:output:0;token_and_position_embedding_8/strided_slice/stack:output:0=token_and_position_embedding_8/strided_slice/stack_1:output:0=token_and_position_embedding_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*token_and_position_embedding_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : l
*token_and_position_embedding_8/range/limitConst*
_output_shapes
: *
dtype0*
value	B :	l
*token_and_position_embedding_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
$token_and_position_embedding_8/rangeRange3token_and_position_embedding_8/range/start:output:03token_and_position_embedding_8/range/limit:output:03token_and_position_embedding_8/range/delta:output:0*
_output_shapes
:	�
<token_and_position_embedding_8/embedding_17/embedding_lookupResourceGatherDtoken_and_position_embedding_8_embedding_17_embedding_lookup_9510222-token_and_position_embedding_8/range:output:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_17/embedding_lookup/9510222*
_output_shapes

:	*
dtype0�
Etoken_and_position_embedding_8/embedding_17/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_8/embedding_17/embedding_lookup:output:0*
T0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_17/embedding_lookup/9510222*
_output_shapes

:	�
Gtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	�
0token_and_position_embedding_8/embedding_16/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������	�
<token_and_position_embedding_8/embedding_16/embedding_lookupResourceGatherDtoken_and_position_embedding_8_embedding_16_embedding_lookup_95102284token_and_position_embedding_8/embedding_16/Cast:y:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_16/embedding_lookup/9510228*+
_output_shapes
:���������	*
dtype0�
Etoken_and_position_embedding_8/embedding_16/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_8/embedding_16/embedding_lookup:output:0*
T0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_16/embedding_lookup/9510228*+
_output_shapes
:���������	�
Gtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
"token_and_position_embedding_8/addAddV2Ptoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1:output:0Ptoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	
5batch_normalization_34/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_34/moments/meanMeandense_60/Relu:activations:0>batch_normalization_34/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
+batch_normalization_34/moments/StopGradientStopGradient,batch_normalization_34/moments/mean:output:0*
T0*
_output_shapes

: �
0batch_normalization_34/moments/SquaredDifferenceSquaredDifferencedense_60/Relu:activations:04batch_normalization_34/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
9batch_normalization_34/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_34/moments/varianceMean4batch_normalization_34/moments/SquaredDifference:z:0Bbatch_normalization_34/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
&batch_normalization_34/moments/SqueezeSqueeze,batch_normalization_34/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
(batch_normalization_34/moments/Squeeze_1Squeeze0batch_normalization_34/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 q
,batch_normalization_34/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_34/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_34_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
*batch_normalization_34/AssignMovingAvg/subSub=batch_normalization_34/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_34/moments/Squeeze:output:0*
T0*
_output_shapes
: �
*batch_normalization_34/AssignMovingAvg/mulMul.batch_normalization_34/AssignMovingAvg/sub:z:05batch_normalization_34/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
&batch_normalization_34/AssignMovingAvgAssignSubVariableOp>batch_normalization_34_assignmovingavg_readvariableop_resource.batch_normalization_34/AssignMovingAvg/mul:z:06^batch_normalization_34/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_34/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_34/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_34_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
,batch_normalization_34/AssignMovingAvg_1/subSub?batch_normalization_34/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_34/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
,batch_normalization_34/AssignMovingAvg_1/mulMul0batch_normalization_34/AssignMovingAvg_1/sub:z:07batch_normalization_34/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
(batch_normalization_34/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_34_assignmovingavg_1_readvariableop_resource0batch_normalization_34/AssignMovingAvg_1/mul:z:08^batch_normalization_34/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_34/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_34/batchnorm/addAddV21batch_normalization_34/moments/Squeeze_1:output:0/batch_normalization_34/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_34/batchnorm/RsqrtRsqrt(batch_normalization_34/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_34/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_34_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_34/batchnorm/mulMul*batch_normalization_34/batchnorm/Rsqrt:y:0;batch_normalization_34/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_34/batchnorm/mul_1Muldense_60/Relu:activations:0(batch_normalization_34/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
&batch_normalization_34/batchnorm/mul_2Mul/batch_normalization_34/moments/Squeeze:output:0(batch_normalization_34/batchnorm/mul:z:0*
T0*
_output_shapes
: �
/batch_normalization_34/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_34_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_34/batchnorm/subSub7batch_normalization_34/batchnorm/ReadVariableOp:value:0*batch_normalization_34/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_34/batchnorm/add_1AddV2*batch_normalization_34/batchnorm/mul_1:z:0(batch_normalization_34/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
@transformer_block_8/attention/query/einsum/Einsum/ReadVariableOpReadVariableOpItransformer_block_8_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
1transformer_block_8/attention/query/einsum/EinsumEinsum&token_and_position_embedding_8/add:z:0Htransformer_block_8/attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
6transformer_block_8/attention/query/add/ReadVariableOpReadVariableOp?transformer_block_8_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
'transformer_block_8/attention/query/addAddV2:transformer_block_8/attention/query/einsum/Einsum:output:0>transformer_block_8/attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
>transformer_block_8/attention/key/einsum/Einsum/ReadVariableOpReadVariableOpGtransformer_block_8_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
/transformer_block_8/attention/key/einsum/EinsumEinsum&token_and_position_embedding_8/add:z:0Ftransformer_block_8/attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
4transformer_block_8/attention/key/add/ReadVariableOpReadVariableOp=transformer_block_8_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
%transformer_block_8/attention/key/addAddV28transformer_block_8/attention/key/einsum/Einsum:output:0<transformer_block_8/attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
@transformer_block_8/attention/value/einsum/Einsum/ReadVariableOpReadVariableOpItransformer_block_8_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
1transformer_block_8/attention/value/einsum/EinsumEinsum&token_and_position_embedding_8/add:z:0Htransformer_block_8/attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
6transformer_block_8/attention/value/add/ReadVariableOpReadVariableOp?transformer_block_8_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
'transformer_block_8/attention/value/addAddV2:transformer_block_8/attention/value/einsum/Einsum:output:0>transformer_block_8/attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	h
#transformer_block_8/attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
!transformer_block_8/attention/MulMul+transformer_block_8/attention/query/add:z:0,transformer_block_8/attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
+transformer_block_8/attention/einsum/EinsumEinsum)transformer_block_8/attention/key/add:z:0%transformer_block_8/attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
-transformer_block_8/attention/softmax/SoftmaxSoftmax4transformer_block_8/attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
-transformer_block_8/attention/einsum_1/EinsumEinsum7transformer_block_8/attention/softmax/Softmax:softmax:0+transformer_block_8/attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
Ktransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_8_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
<transformer_block_8/attention/attention_output/einsum/EinsumEinsum6transformer_block_8/attention/einsum_1/Einsum:output:0Stransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
Atransformer_block_8/attention/attention_output/add/ReadVariableOpReadVariableOpJtransformer_block_8_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
2transformer_block_8/attention/attention_output/addAddV2Etransformer_block_8/attention/attention_output/einsum/Einsum:output:0Itransformer_block_8/attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	q
,transformer_block_8/dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
*transformer_block_8/dropout_48/dropout/MulMul6transformer_block_8/attention/attention_output/add:z:05transformer_block_8/dropout_48/dropout/Const:output:0*
T0*+
_output_shapes
:���������	�
,transformer_block_8/dropout_48/dropout/ShapeShape6transformer_block_8/attention/attention_output/add:z:0*
T0*
_output_shapes
:�
Ctransformer_block_8/dropout_48/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_8/dropout_48/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0z
5transformer_block_8/dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
3transformer_block_8/dropout_48/dropout/GreaterEqualGreaterEqualLtransformer_block_8/dropout_48/dropout/random_uniform/RandomUniform:output:0>transformer_block_8/dropout_48/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
+transformer_block_8/dropout_48/dropout/CastCast7transformer_block_8/dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
,transformer_block_8/dropout_48/dropout/Mul_1Mul.transformer_block_8/dropout_48/dropout/Mul:z:0/transformer_block_8/dropout_48/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	�
transformer_block_8/addAddV2&token_and_position_embedding_8/add:z:00transformer_block_8/dropout_48/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	�
Itransformer_block_8/layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_block_8/layer_normalization_16/moments/meanMeantransformer_block_8/add:z:0Rtransformer_block_8/layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
?transformer_block_8/layer_normalization_16/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Dtransformer_block_8/layer_normalization_16/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add:z:0Htransformer_block_8/layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Mtransformer_block_8/layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_block_8/layer_normalization_16/moments/varianceMeanHtransformer_block_8/layer_normalization_16/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(
:transformer_block_8/layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
8transformer_block_8/layer_normalization_16/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_16/moments/variance:output:0Ctransformer_block_8/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_16/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
8transformer_block_8/layer_normalization_16/batchnorm/mulMul>transformer_block_8/layer_normalization_16/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_16/batchnorm/mul_1Multransformer_block_8/add:z:0<transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_16/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_16/moments/mean:output:0<transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
8transformer_block_8/layer_normalization_16/batchnorm/subSubKtransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_16/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_16/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
Btransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_8_sequential_8_dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
8transformer_block_8/sequential_8/dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block_8/sequential_8/dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
9transformer_block_8/sequential_8/dense_56/Tensordot/ShapeShape>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Atransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2GatherV2Btransformer_block_8/sequential_8/dense_56/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_56/Tensordot/free:output:0Jtransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ctransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1GatherV2Btransformer_block_8/sequential_8/dense_56/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_56/Tensordot/axes:output:0Ltransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
9transformer_block_8/sequential_8/dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
8transformer_block_8/sequential_8/dense_56/Tensordot/ProdProdEtransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2:output:0Btransformer_block_8/sequential_8/dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: �
;transformer_block_8/sequential_8/dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
:transformer_block_8/sequential_8/dense_56/Tensordot/Prod_1ProdGtransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1:output:0Dtransformer_block_8/sequential_8/dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
?transformer_block_8/sequential_8/dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:transformer_block_8/sequential_8/dense_56/Tensordot/concatConcatV2Atransformer_block_8/sequential_8/dense_56/Tensordot/free:output:0Atransformer_block_8/sequential_8/dense_56/Tensordot/axes:output:0Htransformer_block_8/sequential_8/dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
9transformer_block_8/sequential_8/dense_56/Tensordot/stackPackAtransformer_block_8/sequential_8/dense_56/Tensordot/Prod:output:0Ctransformer_block_8/sequential_8/dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
=transformer_block_8/sequential_8/dense_56/Tensordot/transpose	Transpose>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0Ctransformer_block_8/sequential_8/dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
;transformer_block_8/sequential_8/dense_56/Tensordot/ReshapeReshapeAtransformer_block_8/sequential_8/dense_56/Tensordot/transpose:y:0Btransformer_block_8/sequential_8/dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
:transformer_block_8/sequential_8/dense_56/Tensordot/MatMulMatMulDtransformer_block_8/sequential_8/dense_56/Tensordot/Reshape:output:0Jtransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;transformer_block_8/sequential_8/dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_block_8/sequential_8/dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<transformer_block_8/sequential_8/dense_56/Tensordot/concat_1ConcatV2Etransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2:output:0Dtransformer_block_8/sequential_8/dense_56/Tensordot/Const_2:output:0Jtransformer_block_8/sequential_8/dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
3transformer_block_8/sequential_8/dense_56/TensordotReshapeDtransformer_block_8/sequential_8/dense_56/Tensordot/MatMul:product:0Etransformer_block_8/sequential_8/dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
@transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_8_sequential_8_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
1transformer_block_8/sequential_8/dense_56/BiasAddBiasAdd<transformer_block_8/sequential_8/dense_56/Tensordot:output:0Htransformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
.transformer_block_8/sequential_8/dense_56/ReluRelu:transformer_block_8/sequential_8/dense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
Btransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_8_sequential_8_dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
8transformer_block_8/sequential_8/dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block_8/sequential_8/dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
9transformer_block_8/sequential_8/dense_57/Tensordot/ShapeShape<transformer_block_8/sequential_8/dense_56/Relu:activations:0*
T0*
_output_shapes
:�
Atransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2GatherV2Btransformer_block_8/sequential_8/dense_57/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_57/Tensordot/free:output:0Jtransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ctransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1GatherV2Btransformer_block_8/sequential_8/dense_57/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_57/Tensordot/axes:output:0Ltransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
9transformer_block_8/sequential_8/dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
8transformer_block_8/sequential_8/dense_57/Tensordot/ProdProdEtransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2:output:0Btransformer_block_8/sequential_8/dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: �
;transformer_block_8/sequential_8/dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
:transformer_block_8/sequential_8/dense_57/Tensordot/Prod_1ProdGtransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1:output:0Dtransformer_block_8/sequential_8/dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
?transformer_block_8/sequential_8/dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:transformer_block_8/sequential_8/dense_57/Tensordot/concatConcatV2Atransformer_block_8/sequential_8/dense_57/Tensordot/free:output:0Atransformer_block_8/sequential_8/dense_57/Tensordot/axes:output:0Htransformer_block_8/sequential_8/dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
9transformer_block_8/sequential_8/dense_57/Tensordot/stackPackAtransformer_block_8/sequential_8/dense_57/Tensordot/Prod:output:0Ctransformer_block_8/sequential_8/dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
=transformer_block_8/sequential_8/dense_57/Tensordot/transpose	Transpose<transformer_block_8/sequential_8/dense_56/Relu:activations:0Ctransformer_block_8/sequential_8/dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
;transformer_block_8/sequential_8/dense_57/Tensordot/ReshapeReshapeAtransformer_block_8/sequential_8/dense_57/Tensordot/transpose:y:0Btransformer_block_8/sequential_8/dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
:transformer_block_8/sequential_8/dense_57/Tensordot/MatMulMatMulDtransformer_block_8/sequential_8/dense_57/Tensordot/Reshape:output:0Jtransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;transformer_block_8/sequential_8/dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_block_8/sequential_8/dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<transformer_block_8/sequential_8/dense_57/Tensordot/concat_1ConcatV2Etransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2:output:0Dtransformer_block_8/sequential_8/dense_57/Tensordot/Const_2:output:0Jtransformer_block_8/sequential_8/dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
3transformer_block_8/sequential_8/dense_57/TensordotReshapeDtransformer_block_8/sequential_8/dense_57/Tensordot/MatMul:product:0Etransformer_block_8/sequential_8/dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
@transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_8_sequential_8_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
1transformer_block_8/sequential_8/dense_57/BiasAddBiasAdd<transformer_block_8/sequential_8/dense_57/Tensordot:output:0Htransformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	q
,transformer_block_8/dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
*transformer_block_8/dropout_49/dropout/MulMul:transformer_block_8/sequential_8/dense_57/BiasAdd:output:05transformer_block_8/dropout_49/dropout/Const:output:0*
T0*+
_output_shapes
:���������	�
,transformer_block_8/dropout_49/dropout/ShapeShape:transformer_block_8/sequential_8/dense_57/BiasAdd:output:0*
T0*
_output_shapes
:�
Ctransformer_block_8/dropout_49/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_8/dropout_49/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0z
5transformer_block_8/dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
3transformer_block_8/dropout_49/dropout/GreaterEqualGreaterEqualLtransformer_block_8/dropout_49/dropout/random_uniform/RandomUniform:output:0>transformer_block_8/dropout_49/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
+transformer_block_8/dropout_49/dropout/CastCast7transformer_block_8/dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
,transformer_block_8/dropout_49/dropout/Mul_1Mul.transformer_block_8/dropout_49/dropout/Mul:z:0/transformer_block_8/dropout_49/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	�
transformer_block_8/add_1AddV2>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:00transformer_block_8/dropout_49/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	�
Itransformer_block_8/layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_block_8/layer_normalization_17/moments/meanMeantransformer_block_8/add_1:z:0Rtransformer_block_8/layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
?transformer_block_8/layer_normalization_17/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Dtransformer_block_8/layer_normalization_17/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add_1:z:0Htransformer_block_8/layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Mtransformer_block_8/layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_block_8/layer_normalization_17/moments/varianceMeanHtransformer_block_8/layer_normalization_17/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(
:transformer_block_8/layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
8transformer_block_8/layer_normalization_17/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_17/moments/variance:output:0Ctransformer_block_8/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_17/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
8transformer_block_8/layer_normalization_17/batchnorm/mulMul>transformer_block_8/layer_normalization_17/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_17/batchnorm/mul_1Multransformer_block_8/add_1:z:0<transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_17/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_17/moments/mean:output:0<transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
8transformer_block_8/layer_normalization_17/batchnorm/subSubKtransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_17/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_17/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	]
dropout_52/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_52/dropout/MulMul*batch_normalization_34/batchnorm/add_1:z:0!dropout_52/dropout/Const:output:0*
T0*'
_output_shapes
:��������� r
dropout_52/dropout/ShapeShape*batch_normalization_34/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_52/dropout/random_uniform/RandomUniformRandomUniform!dropout_52/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0f
!dropout_52/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_52/dropout/GreaterEqualGreaterEqual8dropout_52/dropout/random_uniform/RandomUniform:output:0*dropout_52/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
dropout_52/dropout/CastCast#dropout_52/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
dropout_52/dropout/Mul_1Muldropout_52/dropout/Mul:z:0dropout_52/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� s
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d_8/MeanMean>transformer_block_8/layer_normalization_17/batchnorm/add_1:z:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_61/MatMulMatMuldropout_52/dropout/Mul_1:z:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_58/MatMulMatMul(global_average_pooling1d_8/Mean:output:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 
5batch_normalization_35/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_35/moments/meanMeandense_61/Relu:activations:0>batch_normalization_35/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
+batch_normalization_35/moments/StopGradientStopGradient,batch_normalization_35/moments/mean:output:0*
T0*
_output_shapes

: �
0batch_normalization_35/moments/SquaredDifferenceSquaredDifferencedense_61/Relu:activations:04batch_normalization_35/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
9batch_normalization_35/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_35/moments/varianceMean4batch_normalization_35/moments/SquaredDifference:z:0Bbatch_normalization_35/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
&batch_normalization_35/moments/SqueezeSqueeze,batch_normalization_35/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
(batch_normalization_35/moments/Squeeze_1Squeeze0batch_normalization_35/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 q
,batch_normalization_35/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_35/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_35_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
*batch_normalization_35/AssignMovingAvg/subSub=batch_normalization_35/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_35/moments/Squeeze:output:0*
T0*
_output_shapes
: �
*batch_normalization_35/AssignMovingAvg/mulMul.batch_normalization_35/AssignMovingAvg/sub:z:05batch_normalization_35/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
&batch_normalization_35/AssignMovingAvgAssignSubVariableOp>batch_normalization_35_assignmovingavg_readvariableop_resource.batch_normalization_35/AssignMovingAvg/mul:z:06^batch_normalization_35/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_35/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_35/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_35_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
,batch_normalization_35/AssignMovingAvg_1/subSub?batch_normalization_35/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_35/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
,batch_normalization_35/AssignMovingAvg_1/mulMul0batch_normalization_35/AssignMovingAvg_1/sub:z:07batch_normalization_35/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
(batch_normalization_35/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_35_assignmovingavg_1_readvariableop_resource0batch_normalization_35/AssignMovingAvg_1/mul:z:08^batch_normalization_35/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_35/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_35/batchnorm/addAddV21batch_normalization_35/moments/Squeeze_1:output:0/batch_normalization_35/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_35/batchnorm/RsqrtRsqrt(batch_normalization_35/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_35/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_35_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_35/batchnorm/mulMul*batch_normalization_35/batchnorm/Rsqrt:y:0;batch_normalization_35/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_35/batchnorm/mul_1Muldense_61/Relu:activations:0(batch_normalization_35/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
&batch_normalization_35/batchnorm/mul_2Mul/batch_normalization_35/moments/Squeeze:output:0(batch_normalization_35/batchnorm/mul:z:0*
T0*
_output_shapes
: �
/batch_normalization_35/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_35_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_35/batchnorm/subSub7batch_normalization_35/batchnorm/ReadVariableOp:value:0*batch_normalization_35/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_35/batchnorm/add_1AddV2*batch_normalization_35/batchnorm/mul_1:z:0(batch_normalization_35/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� 
5batch_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_32/moments/meanMeandense_58/Relu:activations:0>batch_normalization_32/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
+batch_normalization_32/moments/StopGradientStopGradient,batch_normalization_32/moments/mean:output:0*
T0*
_output_shapes

: �
0batch_normalization_32/moments/SquaredDifferenceSquaredDifferencedense_58/Relu:activations:04batch_normalization_32/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
9batch_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_32/moments/varianceMean4batch_normalization_32/moments/SquaredDifference:z:0Bbatch_normalization_32/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
&batch_normalization_32/moments/SqueezeSqueeze,batch_normalization_32/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
(batch_normalization_32/moments/Squeeze_1Squeeze0batch_normalization_32/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 q
,batch_normalization_32/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_32/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_32_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
*batch_normalization_32/AssignMovingAvg/subSub=batch_normalization_32/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_32/moments/Squeeze:output:0*
T0*
_output_shapes
: �
*batch_normalization_32/AssignMovingAvg/mulMul.batch_normalization_32/AssignMovingAvg/sub:z:05batch_normalization_32/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
&batch_normalization_32/AssignMovingAvgAssignSubVariableOp>batch_normalization_32_assignmovingavg_readvariableop_resource.batch_normalization_32/AssignMovingAvg/mul:z:06^batch_normalization_32/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_32/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_32/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_32_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
,batch_normalization_32/AssignMovingAvg_1/subSub?batch_normalization_32/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_32/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
,batch_normalization_32/AssignMovingAvg_1/mulMul0batch_normalization_32/AssignMovingAvg_1/sub:z:07batch_normalization_32/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
(batch_normalization_32/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_32_assignmovingavg_1_readvariableop_resource0batch_normalization_32/AssignMovingAvg_1/mul:z:08^batch_normalization_32/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_32/batchnorm/addAddV21batch_normalization_32/moments/Squeeze_1:output:0/batch_normalization_32/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_32/batchnorm/RsqrtRsqrt(batch_normalization_32/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_32/batchnorm/mulMul*batch_normalization_32/batchnorm/Rsqrt:y:0;batch_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_32/batchnorm/mul_1Muldense_58/Relu:activations:0(batch_normalization_32/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
&batch_normalization_32/batchnorm/mul_2Mul/batch_normalization_32/moments/Squeeze:output:0(batch_normalization_32/batchnorm/mul:z:0*
T0*
_output_shapes
: �
/batch_normalization_32/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_32/batchnorm/subSub7batch_normalization_32/batchnorm/ReadVariableOp:value:0*batch_normalization_32/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_32/batchnorm/add_1AddV2*batch_normalization_32/batchnorm/mul_1:z:0(batch_normalization_32/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� ]
dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_50/dropout/MulMul*batch_normalization_32/batchnorm/add_1:z:0!dropout_50/dropout/Const:output:0*
T0*'
_output_shapes
:��������� r
dropout_50/dropout/ShapeShape*batch_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_50/dropout/random_uniform/RandomUniformRandomUniform!dropout_50/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0f
!dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_50/dropout/GreaterEqualGreaterEqual8dropout_50/dropout/random_uniform/RandomUniform:output:0*dropout_50/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
dropout_50/dropout/CastCast#dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
dropout_50/dropout/Mul_1Muldropout_50/dropout/Mul:z:0dropout_50/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� ]
dropout_53/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_53/dropout/MulMul*batch_normalization_35/batchnorm/add_1:z:0!dropout_53/dropout/Const:output:0*
T0*'
_output_shapes
:��������� r
dropout_53/dropout/ShapeShape*batch_normalization_35/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_53/dropout/random_uniform/RandomUniformRandomUniform!dropout_53/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0f
!dropout_53/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_53/dropout/GreaterEqualGreaterEqual8dropout_53/dropout/random_uniform/RandomUniform:output:0*dropout_53/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
dropout_53/dropout/CastCast#dropout_53/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
dropout_53/dropout/Mul_1Muldropout_53/dropout/Mul:z:0dropout_53/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� [
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_8/concatConcatV2dropout_50/dropout/Mul_1:z:0dropout_53/dropout/Mul_1:z:0"concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_62/MatMulMatMulconcatenate_8/concat:output:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_62/SoftmaxSoftmaxdense_62/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_62/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_32/AssignMovingAvg6^batch_normalization_32/AssignMovingAvg/ReadVariableOp)^batch_normalization_32/AssignMovingAvg_18^batch_normalization_32/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_32/batchnorm/ReadVariableOp4^batch_normalization_32/batchnorm/mul/ReadVariableOp'^batch_normalization_33/AssignMovingAvg6^batch_normalization_33/AssignMovingAvg/ReadVariableOp)^batch_normalization_33/AssignMovingAvg_18^batch_normalization_33/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_33/batchnorm/ReadVariableOp4^batch_normalization_33/batchnorm/mul/ReadVariableOp'^batch_normalization_34/AssignMovingAvg6^batch_normalization_34/AssignMovingAvg/ReadVariableOp)^batch_normalization_34/AssignMovingAvg_18^batch_normalization_34/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_34/batchnorm/ReadVariableOp4^batch_normalization_34/batchnorm/mul/ReadVariableOp'^batch_normalization_35/AssignMovingAvg6^batch_normalization_35/AssignMovingAvg/ReadVariableOp)^batch_normalization_35/AssignMovingAvg_18^batch_normalization_35/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_35/batchnorm/ReadVariableOp4^batch_normalization_35/batchnorm/mul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp=^token_and_position_embedding_8/embedding_16/embedding_lookup=^token_and_position_embedding_8/embedding_17/embedding_lookupB^transformer_block_8/attention/attention_output/add/ReadVariableOpL^transformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp5^transformer_block_8/attention/key/add/ReadVariableOp?^transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp7^transformer_block_8/attention/query/add/ReadVariableOpA^transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp7^transformer_block_8/attention/value/add/ReadVariableOpA^transformer_block_8/attention/value/einsum/Einsum/ReadVariableOpD^transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpD^transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpA^transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOpC^transformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpA^transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOpC^transformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_32/AssignMovingAvg&batch_normalization_32/AssignMovingAvg2n
5batch_normalization_32/AssignMovingAvg/ReadVariableOp5batch_normalization_32/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_32/AssignMovingAvg_1(batch_normalization_32/AssignMovingAvg_12r
7batch_normalization_32/AssignMovingAvg_1/ReadVariableOp7batch_normalization_32/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_32/batchnorm/ReadVariableOp/batch_normalization_32/batchnorm/ReadVariableOp2j
3batch_normalization_32/batchnorm/mul/ReadVariableOp3batch_normalization_32/batchnorm/mul/ReadVariableOp2P
&batch_normalization_33/AssignMovingAvg&batch_normalization_33/AssignMovingAvg2n
5batch_normalization_33/AssignMovingAvg/ReadVariableOp5batch_normalization_33/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_33/AssignMovingAvg_1(batch_normalization_33/AssignMovingAvg_12r
7batch_normalization_33/AssignMovingAvg_1/ReadVariableOp7batch_normalization_33/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_33/batchnorm/ReadVariableOp/batch_normalization_33/batchnorm/ReadVariableOp2j
3batch_normalization_33/batchnorm/mul/ReadVariableOp3batch_normalization_33/batchnorm/mul/ReadVariableOp2P
&batch_normalization_34/AssignMovingAvg&batch_normalization_34/AssignMovingAvg2n
5batch_normalization_34/AssignMovingAvg/ReadVariableOp5batch_normalization_34/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_34/AssignMovingAvg_1(batch_normalization_34/AssignMovingAvg_12r
7batch_normalization_34/AssignMovingAvg_1/ReadVariableOp7batch_normalization_34/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_34/batchnorm/ReadVariableOp/batch_normalization_34/batchnorm/ReadVariableOp2j
3batch_normalization_34/batchnorm/mul/ReadVariableOp3batch_normalization_34/batchnorm/mul/ReadVariableOp2P
&batch_normalization_35/AssignMovingAvg&batch_normalization_35/AssignMovingAvg2n
5batch_normalization_35/AssignMovingAvg/ReadVariableOp5batch_normalization_35/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_35/AssignMovingAvg_1(batch_normalization_35/AssignMovingAvg_12r
7batch_normalization_35/AssignMovingAvg_1/ReadVariableOp7batch_normalization_35/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_35/batchnorm/ReadVariableOp/batch_normalization_35/batchnorm/ReadVariableOp2j
3batch_normalization_35/batchnorm/mul/ReadVariableOp3batch_normalization_35/batchnorm/mul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2|
<token_and_position_embedding_8/embedding_16/embedding_lookup<token_and_position_embedding_8/embedding_16/embedding_lookup2|
<token_and_position_embedding_8/embedding_17/embedding_lookup<token_and_position_embedding_8/embedding_17/embedding_lookup2�
Atransformer_block_8/attention/attention_output/add/ReadVariableOpAtransformer_block_8/attention/attention_output/add/ReadVariableOp2�
Ktransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOpKtransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp2l
4transformer_block_8/attention/key/add/ReadVariableOp4transformer_block_8/attention/key/add/ReadVariableOp2�
>transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp>transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp2p
6transformer_block_8/attention/query/add/ReadVariableOp6transformer_block_8/attention/query/add/ReadVariableOp2�
@transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp@transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp2p
6transformer_block_8/attention/value/add/ReadVariableOp6transformer_block_8/attention/value/add/ReadVariableOp2�
@transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp@transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp2�
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp2�
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp2�
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp2�
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp2�
@transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp@transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp2�
Btransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpBtransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp2�
@transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp@transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp2�
Btransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOpBtransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp:Q M
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
�
e
,__inference_dropout_51_layer_call_fn_9510716

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
G__inference_dropout_51_layer_call_and_return_conditional_losses_9509082o
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
�
�

%__inference_signature_wrapper_9510606
input_17
input_18
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
StatefulPartitionedCallStatefulPartitionedCallinput_17input_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *+
f&R$
"__inference__wrapped_model_9507735o
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
input_17:QM
'
_output_shapes
:���������

"
_user_specified_name
input_18
�%
�
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9508081

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
�
�

)__inference_model_8_layer_call_fn_9509487
input_17
input_18
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
StatefulPartitionedCallStatefulPartitionedCallinput_17input_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *M
fHRF
D__inference_model_8_layer_call_and_return_conditional_losses_9509302o
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
input_17:QM
'
_output_shapes
:���������

"
_user_specified_name
input_18
�
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507898

inputs"
dense_56_9507856:
dense_56_9507858:"
dense_57_9507892:
dense_57_9507894:
identity�� dense_56/StatefulPartitionedCall� dense_57/StatefulPartitionedCall�
 dense_56/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_9507856dense_56_9507858*
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
E__inference_dense_56_layer_call_and_return_conditional_losses_9507855�
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_9507892dense_57_9507894*
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
E__inference_dense_57_layer_call_and_return_conditional_losses_9507891|
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
*__inference_dense_58_layer_call_fn_9511255

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
E__inference_dense_58_layer_call_and_return_conditional_losses_9508565o
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
��
�
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9511128

inputsK
5attention_query_einsum_einsum_readvariableop_resource:=
+attention_query_add_readvariableop_resource:I
3attention_key_einsum_einsum_readvariableop_resource:;
)attention_key_add_readvariableop_resource:K
5attention_value_einsum_einsum_readvariableop_resource:=
+attention_value_add_readvariableop_resource:V
@attention_attention_output_einsum_einsum_readvariableop_resource:D
6attention_attention_output_add_readvariableop_resource:J
<layer_normalization_16_batchnorm_mul_readvariableop_resource:F
8layer_normalization_16_batchnorm_readvariableop_resource:I
7sequential_8_dense_56_tensordot_readvariableop_resource:C
5sequential_8_dense_56_biasadd_readvariableop_resource:I
7sequential_8_dense_57_tensordot_readvariableop_resource:C
5sequential_8_dense_57_biasadd_readvariableop_resource:J
<layer_normalization_17_batchnorm_mul_readvariableop_resource:F
8layer_normalization_17_batchnorm_readvariableop_resource:
identity��-attention/attention_output/add/ReadVariableOp�7attention/attention_output/einsum/Einsum/ReadVariableOp� attention/key/add/ReadVariableOp�*attention/key/einsum/Einsum/ReadVariableOp�"attention/query/add/ReadVariableOp�,attention/query/einsum/Einsum/ReadVariableOp�"attention/value/add/ReadVariableOp�,attention/value/einsum/Einsum/ReadVariableOp�/layer_normalization_16/batchnorm/ReadVariableOp�3layer_normalization_16/batchnorm/mul/ReadVariableOp�/layer_normalization_17/batchnorm/ReadVariableOp�3layer_normalization_17/batchnorm/mul/ReadVariableOp�,sequential_8/dense_56/BiasAdd/ReadVariableOp�.sequential_8/dense_56/Tensordot/ReadVariableOp�,sequential_8/dense_57/BiasAdd/ReadVariableOp�.sequential_8/dense_57/Tensordot/ReadVariableOp�
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
:���������	]
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_48/dropout/MulMul"attention/attention_output/add:z:0!dropout_48/dropout/Const:output:0*
T0*+
_output_shapes
:���������	j
dropout_48/dropout/ShapeShape"attention/attention_output/add:z:0*
T0*
_output_shapes
:�
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0f
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	h
addAddV2inputsdropout_48/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_16/moments/meanMeanadd:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(k
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/mul_1Muladd:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
.sequential_8/dense_56/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_8/dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_8/dense_56/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_8/dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_56/Tensordot/GatherV2GatherV2.sequential_8/dense_56/Tensordot/Shape:output:0-sequential_8/dense_56/Tensordot/free:output:06sequential_8/dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_8/dense_56/Tensordot/GatherV2_1GatherV2.sequential_8/dense_56/Tensordot/Shape:output:0-sequential_8/dense_56/Tensordot/axes:output:08sequential_8/dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_8/dense_56/Tensordot/ProdProd1sequential_8/dense_56/Tensordot/GatherV2:output:0.sequential_8/dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_8/dense_56/Tensordot/Prod_1Prod3sequential_8/dense_56/Tensordot/GatherV2_1:output:00sequential_8/dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_8/dense_56/Tensordot/concatConcatV2-sequential_8/dense_56/Tensordot/free:output:0-sequential_8/dense_56/Tensordot/axes:output:04sequential_8/dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_8/dense_56/Tensordot/stackPack-sequential_8/dense_56/Tensordot/Prod:output:0/sequential_8/dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_8/dense_56/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_8/dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
'sequential_8/dense_56/Tensordot/ReshapeReshape-sequential_8/dense_56/Tensordot/transpose:y:0.sequential_8/dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_8/dense_56/Tensordot/MatMulMatMul0sequential_8/dense_56/Tensordot/Reshape:output:06sequential_8/dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
'sequential_8/dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_8/dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_56/Tensordot/concat_1ConcatV21sequential_8/dense_56/Tensordot/GatherV2:output:00sequential_8/dense_56/Tensordot/Const_2:output:06sequential_8/dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_8/dense_56/TensordotReshape0sequential_8/dense_56/Tensordot/MatMul:product:01sequential_8/dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
,sequential_8/dense_56/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_8/dense_56/BiasAddBiasAdd(sequential_8/dense_56/Tensordot:output:04sequential_8/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
sequential_8/dense_56/ReluRelu&sequential_8/dense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
.sequential_8/dense_57/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_8/dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_8/dense_57/Tensordot/ShapeShape(sequential_8/dense_56/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_8/dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_57/Tensordot/GatherV2GatherV2.sequential_8/dense_57/Tensordot/Shape:output:0-sequential_8/dense_57/Tensordot/free:output:06sequential_8/dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_8/dense_57/Tensordot/GatherV2_1GatherV2.sequential_8/dense_57/Tensordot/Shape:output:0-sequential_8/dense_57/Tensordot/axes:output:08sequential_8/dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_8/dense_57/Tensordot/ProdProd1sequential_8/dense_57/Tensordot/GatherV2:output:0.sequential_8/dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_8/dense_57/Tensordot/Prod_1Prod3sequential_8/dense_57/Tensordot/GatherV2_1:output:00sequential_8/dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_8/dense_57/Tensordot/concatConcatV2-sequential_8/dense_57/Tensordot/free:output:0-sequential_8/dense_57/Tensordot/axes:output:04sequential_8/dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_8/dense_57/Tensordot/stackPack-sequential_8/dense_57/Tensordot/Prod:output:0/sequential_8/dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_8/dense_57/Tensordot/transpose	Transpose(sequential_8/dense_56/Relu:activations:0/sequential_8/dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
'sequential_8/dense_57/Tensordot/ReshapeReshape-sequential_8/dense_57/Tensordot/transpose:y:0.sequential_8/dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_8/dense_57/Tensordot/MatMulMatMul0sequential_8/dense_57/Tensordot/Reshape:output:06sequential_8/dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
'sequential_8/dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_8/dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_57/Tensordot/concat_1ConcatV21sequential_8/dense_57/Tensordot/GatherV2:output:00sequential_8/dense_57/Tensordot/Const_2:output:06sequential_8/dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_8/dense_57/TensordotReshape0sequential_8/dense_57/Tensordot/MatMul:product:01sequential_8/dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
,sequential_8/dense_57/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_8/dense_57/BiasAddBiasAdd(sequential_8/dense_57/Tensordot:output:04sequential_8/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	]
dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_49/dropout/MulMul&sequential_8/dense_57/BiasAdd:output:0!dropout_49/dropout/Const:output:0*
T0*+
_output_shapes
:���������	n
dropout_49/dropout/ShapeShape&sequential_8/dense_57/BiasAdd:output:0*
T0*
_output_shapes
:�
/dropout_49/dropout/random_uniform/RandomUniformRandomUniform!dropout_49/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0f
!dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_49/dropout/GreaterEqualGreaterEqual8dropout_49/dropout/random_uniform/RandomUniform:output:0*dropout_49/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
dropout_49/dropout/CastCast#dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
dropout_49/dropout/Mul_1Muldropout_49/dropout/Mul:z:0dropout_49/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	�
add_1AddV2*layer_normalization_16/batchnorm/add_1:z:0dropout_49/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_17/moments/meanMean	add_1:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
0layer_normalization_17/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(k
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	}
IdentityIdentity*layer_normalization_17/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp-^sequential_8/dense_56/BiasAdd/ReadVariableOp/^sequential_8/dense_56/Tensordot/ReadVariableOp-^sequential_8/dense_57/BiasAdd/ReadVariableOp/^sequential_8/dense_57/Tensordot/ReadVariableOp*"
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
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2\
,sequential_8/dense_56/BiasAdd/ReadVariableOp,sequential_8/dense_56/BiasAdd/ReadVariableOp2`
.sequential_8/dense_56/Tensordot/ReadVariableOp.sequential_8/dense_56/Tensordot/ReadVariableOp2\
,sequential_8/dense_57/BiasAdd/ReadVariableOp,sequential_8/dense_57/BiasAdd/ReadVariableOp2`
.sequential_8/dense_57/Tensordot/ReadVariableOp.sequential_8/dense_57/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
.__inference_sequential_8_layer_call_fn_9511559

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
GPU 2J 8� *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507958s
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
�

�
E__inference_dense_59_layer_call_and_return_conditional_losses_9510626

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
8__inference_batch_normalization_33_layer_call_fn_9510639

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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9507759o
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
�
�
*__inference_dense_59_layer_call_fn_9510615

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
GPU 2J 8� *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_9508289o
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
�
�
*__inference_dense_57_layer_call_fn_9511722

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
E__inference_dense_57_layer_call_and_return_conditional_losses_9507891s
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
�	
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_9508781

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
�
�
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9511174

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

�
E__inference_dense_58_layer_call_and_return_conditional_losses_9508565

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
�

�
E__inference_dense_59_layer_call_and_return_conditional_losses_9508289

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
�
�
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9510672

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
H
,__inference_dropout_52_layer_call_fn_9511224

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
G__inference_dropout_52_layer_call_and_return_conditional_losses_9508534`
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
�
v
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9511513
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
�	
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_9511473

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
��
�Y
#__inference__traced_restore_9512524
file_prefix2
 assignvariableop_dense_59_kernel:
 .
 assignvariableop_1_dense_59_bias: =
/assignvariableop_2_batch_normalization_33_gamma: <
.assignvariableop_3_batch_normalization_33_beta: C
5assignvariableop_4_batch_normalization_33_moving_mean: G
9assignvariableop_5_batch_normalization_33_moving_variance: 4
"assignvariableop_6_dense_60_kernel:  .
 assignvariableop_7_dense_60_bias: =
/assignvariableop_8_batch_normalization_34_gamma: <
.assignvariableop_9_batch_normalization_34_beta: D
6assignvariableop_10_batch_normalization_34_moving_mean: H
:assignvariableop_11_batch_normalization_34_moving_variance: 5
#assignvariableop_12_dense_58_kernel: /
!assignvariableop_13_dense_58_bias: 5
#assignvariableop_14_dense_61_kernel:  /
!assignvariableop_15_dense_61_bias: >
0assignvariableop_16_batch_normalization_32_gamma: =
/assignvariableop_17_batch_normalization_32_beta: D
6assignvariableop_18_batch_normalization_32_moving_mean: H
:assignvariableop_19_batch_normalization_32_moving_variance: >
0assignvariableop_20_batch_normalization_35_gamma: =
/assignvariableop_21_batch_normalization_35_beta: D
6assignvariableop_22_batch_normalization_35_moving_mean: H
:assignvariableop_23_batch_normalization_35_moving_variance: 5
#assignvariableop_24_dense_62_kernel:@/
!assignvariableop_25_dense_62_bias:$
assignvariableop_26_beta_1: $
assignvariableop_27_beta_2: #
assignvariableop_28_decay: +
!assignvariableop_29_learning_rate: '
assignvariableop_30_adam_iter:	 \
Jassignvariableop_31_token_and_position_embedding_8_embedding_16_embeddings:\
Jassignvariableop_32_token_and_position_embedding_8_embedding_17_embeddings:	T
>assignvariableop_33_transformer_block_8_attention_query_kernel:N
<assignvariableop_34_transformer_block_8_attention_query_bias:R
<assignvariableop_35_transformer_block_8_attention_key_kernel:L
:assignvariableop_36_transformer_block_8_attention_key_bias:T
>assignvariableop_37_transformer_block_8_attention_value_kernel:N
<assignvariableop_38_transformer_block_8_attention_value_bias:_
Iassignvariableop_39_transformer_block_8_attention_attention_output_kernel:U
Gassignvariableop_40_transformer_block_8_attention_attention_output_bias:5
#assignvariableop_41_dense_56_kernel:/
!assignvariableop_42_dense_56_bias:5
#assignvariableop_43_dense_57_kernel:/
!assignvariableop_44_dense_57_bias:R
Dassignvariableop_45_transformer_block_8_layer_normalization_16_gamma:Q
Cassignvariableop_46_transformer_block_8_layer_normalization_16_beta:R
Dassignvariableop_47_transformer_block_8_layer_normalization_17_gamma:Q
Cassignvariableop_48_transformer_block_8_layer_normalization_17_beta:#
assignvariableop_49_total: #
assignvariableop_50_count: <
*assignvariableop_51_adam_dense_59_kernel_m:
 6
(assignvariableop_52_adam_dense_59_bias_m: E
7assignvariableop_53_adam_batch_normalization_33_gamma_m: D
6assignvariableop_54_adam_batch_normalization_33_beta_m: <
*assignvariableop_55_adam_dense_60_kernel_m:  6
(assignvariableop_56_adam_dense_60_bias_m: E
7assignvariableop_57_adam_batch_normalization_34_gamma_m: D
6assignvariableop_58_adam_batch_normalization_34_beta_m: <
*assignvariableop_59_adam_dense_58_kernel_m: 6
(assignvariableop_60_adam_dense_58_bias_m: <
*assignvariableop_61_adam_dense_61_kernel_m:  6
(assignvariableop_62_adam_dense_61_bias_m: E
7assignvariableop_63_adam_batch_normalization_32_gamma_m: D
6assignvariableop_64_adam_batch_normalization_32_beta_m: E
7assignvariableop_65_adam_batch_normalization_35_gamma_m: D
6assignvariableop_66_adam_batch_normalization_35_beta_m: <
*assignvariableop_67_adam_dense_62_kernel_m:@6
(assignvariableop_68_adam_dense_62_bias_m:c
Qassignvariableop_69_adam_token_and_position_embedding_8_embedding_16_embeddings_m:c
Qassignvariableop_70_adam_token_and_position_embedding_8_embedding_17_embeddings_m:	[
Eassignvariableop_71_adam_transformer_block_8_attention_query_kernel_m:U
Cassignvariableop_72_adam_transformer_block_8_attention_query_bias_m:Y
Cassignvariableop_73_adam_transformer_block_8_attention_key_kernel_m:S
Aassignvariableop_74_adam_transformer_block_8_attention_key_bias_m:[
Eassignvariableop_75_adam_transformer_block_8_attention_value_kernel_m:U
Cassignvariableop_76_adam_transformer_block_8_attention_value_bias_m:f
Passignvariableop_77_adam_transformer_block_8_attention_attention_output_kernel_m:\
Nassignvariableop_78_adam_transformer_block_8_attention_attention_output_bias_m:<
*assignvariableop_79_adam_dense_56_kernel_m:6
(assignvariableop_80_adam_dense_56_bias_m:<
*assignvariableop_81_adam_dense_57_kernel_m:6
(assignvariableop_82_adam_dense_57_bias_m:Y
Kassignvariableop_83_adam_transformer_block_8_layer_normalization_16_gamma_m:X
Jassignvariableop_84_adam_transformer_block_8_layer_normalization_16_beta_m:Y
Kassignvariableop_85_adam_transformer_block_8_layer_normalization_17_gamma_m:X
Jassignvariableop_86_adam_transformer_block_8_layer_normalization_17_beta_m:<
*assignvariableop_87_adam_dense_59_kernel_v:
 6
(assignvariableop_88_adam_dense_59_bias_v: E
7assignvariableop_89_adam_batch_normalization_33_gamma_v: D
6assignvariableop_90_adam_batch_normalization_33_beta_v: <
*assignvariableop_91_adam_dense_60_kernel_v:  6
(assignvariableop_92_adam_dense_60_bias_v: E
7assignvariableop_93_adam_batch_normalization_34_gamma_v: D
6assignvariableop_94_adam_batch_normalization_34_beta_v: <
*assignvariableop_95_adam_dense_58_kernel_v: 6
(assignvariableop_96_adam_dense_58_bias_v: <
*assignvariableop_97_adam_dense_61_kernel_v:  6
(assignvariableop_98_adam_dense_61_bias_v: E
7assignvariableop_99_adam_batch_normalization_32_gamma_v: E
7assignvariableop_100_adam_batch_normalization_32_beta_v: F
8assignvariableop_101_adam_batch_normalization_35_gamma_v: E
7assignvariableop_102_adam_batch_normalization_35_beta_v: =
+assignvariableop_103_adam_dense_62_kernel_v:@7
)assignvariableop_104_adam_dense_62_bias_v:d
Rassignvariableop_105_adam_token_and_position_embedding_8_embedding_16_embeddings_v:d
Rassignvariableop_106_adam_token_and_position_embedding_8_embedding_17_embeddings_v:	\
Fassignvariableop_107_adam_transformer_block_8_attention_query_kernel_v:V
Dassignvariableop_108_adam_transformer_block_8_attention_query_bias_v:Z
Dassignvariableop_109_adam_transformer_block_8_attention_key_kernel_v:T
Bassignvariableop_110_adam_transformer_block_8_attention_key_bias_v:\
Fassignvariableop_111_adam_transformer_block_8_attention_value_kernel_v:V
Dassignvariableop_112_adam_transformer_block_8_attention_value_bias_v:g
Qassignvariableop_113_adam_transformer_block_8_attention_attention_output_kernel_v:]
Oassignvariableop_114_adam_transformer_block_8_attention_attention_output_bias_v:=
+assignvariableop_115_adam_dense_56_kernel_v:7
)assignvariableop_116_adam_dense_56_bias_v:=
+assignvariableop_117_adam_dense_57_kernel_v:7
)assignvariableop_118_adam_dense_57_bias_v:Z
Lassignvariableop_119_adam_transformer_block_8_layer_normalization_16_gamma_v:Y
Kassignvariableop_120_adam_transformer_block_8_layer_normalization_16_beta_v:Z
Lassignvariableop_121_adam_transformer_block_8_layer_normalization_17_gamma_v:Y
Kassignvariableop_122_adam_transformer_block_8_layer_normalization_17_beta_v:
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
AssignVariableOpAssignVariableOp assignvariableop_dense_59_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_59_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_33_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_33_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_33_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_33_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_60_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_60_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_34_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_34_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_34_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_34_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_58_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_58_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_61_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_61_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_32_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_32_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_32_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_32_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_35_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_35_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_35_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_35_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_62_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_62_biasIdentity_25:output:0"/device:CPU:0*
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
AssignVariableOp_31AssignVariableOpJassignvariableop_31_token_and_position_embedding_8_embedding_16_embeddingsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpJassignvariableop_32_token_and_position_embedding_8_embedding_17_embeddingsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp>assignvariableop_33_transformer_block_8_attention_query_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp<assignvariableop_34_transformer_block_8_attention_query_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp<assignvariableop_35_transformer_block_8_attention_key_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp:assignvariableop_36_transformer_block_8_attention_key_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp>assignvariableop_37_transformer_block_8_attention_value_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp<assignvariableop_38_transformer_block_8_attention_value_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpIassignvariableop_39_transformer_block_8_attention_attention_output_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpGassignvariableop_40_transformer_block_8_attention_attention_output_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp#assignvariableop_41_dense_56_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp!assignvariableop_42_dense_56_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp#assignvariableop_43_dense_57_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp!assignvariableop_44_dense_57_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpDassignvariableop_45_transformer_block_8_layer_normalization_16_gammaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpCassignvariableop_46_transformer_block_8_layer_normalization_16_betaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpDassignvariableop_47_transformer_block_8_layer_normalization_17_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpCassignvariableop_48_transformer_block_8_layer_normalization_17_betaIdentity_48:output:0"/device:CPU:0*
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
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_59_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_59_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_33_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_33_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_60_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_60_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_34_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_34_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_58_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_58_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_61_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_61_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_32_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_32_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_batch_normalization_35_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_batch_normalization_35_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_62_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_62_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpQassignvariableop_69_adam_token_and_position_embedding_8_embedding_16_embeddings_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOpQassignvariableop_70_adam_token_and_position_embedding_8_embedding_17_embeddings_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOpEassignvariableop_71_adam_transformer_block_8_attention_query_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOpCassignvariableop_72_adam_transformer_block_8_attention_query_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOpCassignvariableop_73_adam_transformer_block_8_attention_key_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpAassignvariableop_74_adam_transformer_block_8_attention_key_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpEassignvariableop_75_adam_transformer_block_8_attention_value_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpCassignvariableop_76_adam_transformer_block_8_attention_value_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpPassignvariableop_77_adam_transformer_block_8_attention_attention_output_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpNassignvariableop_78_adam_transformer_block_8_attention_attention_output_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_56_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_56_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_57_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_57_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOpKassignvariableop_83_adam_transformer_block_8_layer_normalization_16_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOpJassignvariableop_84_adam_transformer_block_8_layer_normalization_16_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOpKassignvariableop_85_adam_transformer_block_8_layer_normalization_17_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpJassignvariableop_86_adam_transformer_block_8_layer_normalization_17_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_59_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_59_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp7assignvariableop_89_adam_batch_normalization_33_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_batch_normalization_33_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_60_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_60_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp7assignvariableop_93_adam_batch_normalization_34_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp6assignvariableop_94_adam_batch_normalization_34_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_dense_58_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_dense_58_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_dense_61_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_dense_61_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp7assignvariableop_99_adam_batch_normalization_32_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_batch_normalization_32_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp8assignvariableop_101_adam_batch_normalization_35_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp7assignvariableop_102_adam_batch_normalization_35_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_dense_62_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_dense_62_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOpRassignvariableop_105_adam_token_and_position_embedding_8_embedding_16_embeddings_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOpRassignvariableop_106_adam_token_and_position_embedding_8_embedding_17_embeddings_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOpFassignvariableop_107_adam_transformer_block_8_attention_query_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOpDassignvariableop_108_adam_transformer_block_8_attention_query_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOpDassignvariableop_109_adam_transformer_block_8_attention_key_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOpBassignvariableop_110_adam_transformer_block_8_attention_key_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOpFassignvariableop_111_adam_transformer_block_8_attention_value_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOpDassignvariableop_112_adam_transformer_block_8_attention_value_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOpQassignvariableop_113_adam_transformer_block_8_attention_attention_output_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOpOassignvariableop_114_adam_transformer_block_8_attention_attention_output_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp+assignvariableop_115_adam_dense_56_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp)assignvariableop_116_adam_dense_56_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp+assignvariableop_117_adam_dense_57_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp)assignvariableop_118_adam_dense_57_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOpLassignvariableop_119_adam_transformer_block_8_layer_normalization_16_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOpKassignvariableop_120_adam_transformer_block_8_layer_normalization_16_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOpLassignvariableop_121_adam_transformer_block_8_layer_normalization_17_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOpKassignvariableop_122_adam_transformer_block_8_layer_normalization_17_beta_vIdentity_122:output:0"/device:CPU:0*
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
�
e
G__inference_dropout_51_layer_call_and_return_conditional_losses_9510721

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
�%
�
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9507806

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
��
�3
"__inference__wrapped_model_9507735
input_17
input_18A
/model_8_dense_59_matmul_readvariableop_resource:
 >
0model_8_dense_59_biasadd_readvariableop_resource: N
@model_8_batch_normalization_33_batchnorm_readvariableop_resource: R
Dmodel_8_batch_normalization_33_batchnorm_mul_readvariableop_resource: P
Bmodel_8_batch_normalization_33_batchnorm_readvariableop_1_resource: P
Bmodel_8_batch_normalization_33_batchnorm_readvariableop_2_resource: A
/model_8_dense_60_matmul_readvariableop_resource:  >
0model_8_dense_60_biasadd_readvariableop_resource: ^
Lmodel_8_token_and_position_embedding_8_embedding_17_embedding_lookup_9507523:	^
Lmodel_8_token_and_position_embedding_8_embedding_16_embedding_lookup_9507529:N
@model_8_batch_normalization_34_batchnorm_readvariableop_resource: R
Dmodel_8_batch_normalization_34_batchnorm_mul_readvariableop_resource: P
Bmodel_8_batch_normalization_34_batchnorm_readvariableop_1_resource: P
Bmodel_8_batch_normalization_34_batchnorm_readvariableop_2_resource: g
Qmodel_8_transformer_block_8_attention_query_einsum_einsum_readvariableop_resource:Y
Gmodel_8_transformer_block_8_attention_query_add_readvariableop_resource:e
Omodel_8_transformer_block_8_attention_key_einsum_einsum_readvariableop_resource:W
Emodel_8_transformer_block_8_attention_key_add_readvariableop_resource:g
Qmodel_8_transformer_block_8_attention_value_einsum_einsum_readvariableop_resource:Y
Gmodel_8_transformer_block_8_attention_value_add_readvariableop_resource:r
\model_8_transformer_block_8_attention_attention_output_einsum_einsum_readvariableop_resource:`
Rmodel_8_transformer_block_8_attention_attention_output_add_readvariableop_resource:f
Xmodel_8_transformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource:b
Tmodel_8_transformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource:e
Smodel_8_transformer_block_8_sequential_8_dense_56_tensordot_readvariableop_resource:_
Qmodel_8_transformer_block_8_sequential_8_dense_56_biasadd_readvariableop_resource:e
Smodel_8_transformer_block_8_sequential_8_dense_57_tensordot_readvariableop_resource:_
Qmodel_8_transformer_block_8_sequential_8_dense_57_biasadd_readvariableop_resource:f
Xmodel_8_transformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource:b
Tmodel_8_transformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource:A
/model_8_dense_61_matmul_readvariableop_resource:  >
0model_8_dense_61_biasadd_readvariableop_resource: A
/model_8_dense_58_matmul_readvariableop_resource: >
0model_8_dense_58_biasadd_readvariableop_resource: N
@model_8_batch_normalization_35_batchnorm_readvariableop_resource: R
Dmodel_8_batch_normalization_35_batchnorm_mul_readvariableop_resource: P
Bmodel_8_batch_normalization_35_batchnorm_readvariableop_1_resource: P
Bmodel_8_batch_normalization_35_batchnorm_readvariableop_2_resource: N
@model_8_batch_normalization_32_batchnorm_readvariableop_resource: R
Dmodel_8_batch_normalization_32_batchnorm_mul_readvariableop_resource: P
Bmodel_8_batch_normalization_32_batchnorm_readvariableop_1_resource: P
Bmodel_8_batch_normalization_32_batchnorm_readvariableop_2_resource: A
/model_8_dense_62_matmul_readvariableop_resource:@>
0model_8_dense_62_biasadd_readvariableop_resource:
identity��7model_8/batch_normalization_32/batchnorm/ReadVariableOp�9model_8/batch_normalization_32/batchnorm/ReadVariableOp_1�9model_8/batch_normalization_32/batchnorm/ReadVariableOp_2�;model_8/batch_normalization_32/batchnorm/mul/ReadVariableOp�7model_8/batch_normalization_33/batchnorm/ReadVariableOp�9model_8/batch_normalization_33/batchnorm/ReadVariableOp_1�9model_8/batch_normalization_33/batchnorm/ReadVariableOp_2�;model_8/batch_normalization_33/batchnorm/mul/ReadVariableOp�7model_8/batch_normalization_34/batchnorm/ReadVariableOp�9model_8/batch_normalization_34/batchnorm/ReadVariableOp_1�9model_8/batch_normalization_34/batchnorm/ReadVariableOp_2�;model_8/batch_normalization_34/batchnorm/mul/ReadVariableOp�7model_8/batch_normalization_35/batchnorm/ReadVariableOp�9model_8/batch_normalization_35/batchnorm/ReadVariableOp_1�9model_8/batch_normalization_35/batchnorm/ReadVariableOp_2�;model_8/batch_normalization_35/batchnorm/mul/ReadVariableOp�'model_8/dense_58/BiasAdd/ReadVariableOp�&model_8/dense_58/MatMul/ReadVariableOp�'model_8/dense_59/BiasAdd/ReadVariableOp�&model_8/dense_59/MatMul/ReadVariableOp�'model_8/dense_60/BiasAdd/ReadVariableOp�&model_8/dense_60/MatMul/ReadVariableOp�'model_8/dense_61/BiasAdd/ReadVariableOp�&model_8/dense_61/MatMul/ReadVariableOp�'model_8/dense_62/BiasAdd/ReadVariableOp�&model_8/dense_62/MatMul/ReadVariableOp�Dmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup�Dmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup�Imodel_8/transformer_block_8/attention/attention_output/add/ReadVariableOp�Smodel_8/transformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp�<model_8/transformer_block_8/attention/key/add/ReadVariableOp�Fmodel_8/transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp�>model_8/transformer_block_8/attention/query/add/ReadVariableOp�Hmodel_8/transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp�>model_8/transformer_block_8/attention/value/add/ReadVariableOp�Hmodel_8/transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp�Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp�Omodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp�Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp�Omodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp�Hmodel_8/transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp�Jmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp�Hmodel_8/transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp�Jmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp�
&model_8/dense_59/MatMul/ReadVariableOpReadVariableOp/model_8_dense_59_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype0�
model_8/dense_59/MatMulMatMulinput_18.model_8/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_8/dense_59/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_8/dense_59/BiasAddBiasAdd!model_8/dense_59/MatMul:product:0/model_8/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
model_8/dense_59/ReluRelu!model_8/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
7model_8/batch_normalization_33/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0s
.model_8/batch_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,model_8/batch_normalization_33/batchnorm/addAddV2?model_8/batch_normalization_33/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_33/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_33/batchnorm/RsqrtRsqrt0model_8/batch_normalization_33/batchnorm/add:z:0*
T0*
_output_shapes
: �
;model_8/batch_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
,model_8/batch_normalization_33/batchnorm/mulMul2model_8/batch_normalization_33/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_33/batchnorm/mul_1Mul#model_8/dense_59/Relu:activations:00model_8/batch_normalization_33/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
9model_8/batch_normalization_33/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_33_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
.model_8/batch_normalization_33/batchnorm/mul_2MulAmodel_8/batch_normalization_33/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_33/batchnorm/mul:z:0*
T0*
_output_shapes
: �
9model_8/batch_normalization_33/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_33_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
,model_8/batch_normalization_33/batchnorm/subSubAmodel_8/batch_normalization_33/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_33/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_33/batchnorm/add_1AddV22model_8/batch_normalization_33/batchnorm/mul_1:z:00model_8/batch_normalization_33/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
model_8/dropout_51/IdentityIdentity2model_8/batch_normalization_33/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� �
&model_8/dense_60/MatMul/ReadVariableOpReadVariableOp/model_8_dense_60_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_8/dense_60/MatMulMatMul$model_8/dropout_51/Identity:output:0.model_8/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_8/dense_60/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_8/dense_60/BiasAddBiasAdd!model_8/dense_60/MatMul:product:0/model_8/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
model_8/dense_60/ReluRelu!model_8/dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:��������� d
,model_8/token_and_position_embedding_8/ShapeShapeinput_17*
T0*
_output_shapes
:�
:model_8/token_and_position_embedding_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
<model_8/token_and_position_embedding_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
<model_8/token_and_position_embedding_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4model_8/token_and_position_embedding_8/strided_sliceStridedSlice5model_8/token_and_position_embedding_8/Shape:output:0Cmodel_8/token_and_position_embedding_8/strided_slice/stack:output:0Emodel_8/token_and_position_embedding_8/strided_slice/stack_1:output:0Emodel_8/token_and_position_embedding_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2model_8/token_and_position_embedding_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : t
2model_8/token_and_position_embedding_8/range/limitConst*
_output_shapes
: *
dtype0*
value	B :	t
2model_8/token_and_position_embedding_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
,model_8/token_and_position_embedding_8/rangeRange;model_8/token_and_position_embedding_8/range/start:output:0;model_8/token_and_position_embedding_8/range/limit:output:0;model_8/token_and_position_embedding_8/range/delta:output:0*
_output_shapes
:	�
Dmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookupResourceGatherLmodel_8_token_and_position_embedding_8_embedding_17_embedding_lookup_95075235model_8/token_and_position_embedding_8/range:output:0*
Tindices0*_
_classU
SQloc:@model_8/token_and_position_embedding_8/embedding_17/embedding_lookup/9507523*
_output_shapes

:	*
dtype0�
Mmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/IdentityIdentityMmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup:output:0*
T0*_
_classU
SQloc:@model_8/token_and_position_embedding_8/embedding_17/embedding_lookup/9507523*
_output_shapes

:	�
Omodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1IdentityVmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	�
8model_8/token_and_position_embedding_8/embedding_16/CastCastinput_17*

DstT0*

SrcT0*'
_output_shapes
:���������	�
Dmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookupResourceGatherLmodel_8_token_and_position_embedding_8_embedding_16_embedding_lookup_9507529<model_8/token_and_position_embedding_8/embedding_16/Cast:y:0*
Tindices0*_
_classU
SQloc:@model_8/token_and_position_embedding_8/embedding_16/embedding_lookup/9507529*+
_output_shapes
:���������	*
dtype0�
Mmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/IdentityIdentityMmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup:output:0*
T0*_
_classU
SQloc:@model_8/token_and_position_embedding_8/embedding_16/embedding_lookup/9507529*+
_output_shapes
:���������	�
Omodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1IdentityVmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
*model_8/token_and_position_embedding_8/addAddV2Xmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1:output:0Xmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	�
7model_8/batch_normalization_34/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_34_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0s
.model_8/batch_normalization_34/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,model_8/batch_normalization_34/batchnorm/addAddV2?model_8/batch_normalization_34/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_34/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_34/batchnorm/RsqrtRsqrt0model_8/batch_normalization_34/batchnorm/add:z:0*
T0*
_output_shapes
: �
;model_8/batch_normalization_34/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_34_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
,model_8/batch_normalization_34/batchnorm/mulMul2model_8/batch_normalization_34/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_34/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_34/batchnorm/mul_1Mul#model_8/dense_60/Relu:activations:00model_8/batch_normalization_34/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
9model_8/batch_normalization_34/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_34_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
.model_8/batch_normalization_34/batchnorm/mul_2MulAmodel_8/batch_normalization_34/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_34/batchnorm/mul:z:0*
T0*
_output_shapes
: �
9model_8/batch_normalization_34/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_34_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
,model_8/batch_normalization_34/batchnorm/subSubAmodel_8/batch_normalization_34/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_34/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_34/batchnorm/add_1AddV22model_8/batch_normalization_34/batchnorm/mul_1:z:00model_8/batch_normalization_34/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
Hmodel_8/transformer_block_8/attention/query/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_8_transformer_block_8_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
9model_8/transformer_block_8/attention/query/einsum/EinsumEinsum.model_8/token_and_position_embedding_8/add:z:0Pmodel_8/transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
>model_8/transformer_block_8/attention/query/add/ReadVariableOpReadVariableOpGmodel_8_transformer_block_8_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
/model_8/transformer_block_8/attention/query/addAddV2Bmodel_8/transformer_block_8/attention/query/einsum/Einsum:output:0Fmodel_8/transformer_block_8/attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
Fmodel_8/transformer_block_8/attention/key/einsum/Einsum/ReadVariableOpReadVariableOpOmodel_8_transformer_block_8_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
7model_8/transformer_block_8/attention/key/einsum/EinsumEinsum.model_8/token_and_position_embedding_8/add:z:0Nmodel_8/transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
<model_8/transformer_block_8/attention/key/add/ReadVariableOpReadVariableOpEmodel_8_transformer_block_8_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
-model_8/transformer_block_8/attention/key/addAddV2@model_8/transformer_block_8/attention/key/einsum/Einsum:output:0Dmodel_8/transformer_block_8/attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
Hmodel_8/transformer_block_8/attention/value/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_8_transformer_block_8_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
9model_8/transformer_block_8/attention/value/einsum/EinsumEinsum.model_8/token_and_position_embedding_8/add:z:0Pmodel_8/transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
>model_8/transformer_block_8/attention/value/add/ReadVariableOpReadVariableOpGmodel_8_transformer_block_8_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
/model_8/transformer_block_8/attention/value/addAddV2Bmodel_8/transformer_block_8/attention/value/einsum/Einsum:output:0Fmodel_8/transformer_block_8/attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	p
+model_8/transformer_block_8/attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
)model_8/transformer_block_8/attention/MulMul3model_8/transformer_block_8/attention/query/add:z:04model_8/transformer_block_8/attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
3model_8/transformer_block_8/attention/einsum/EinsumEinsum1model_8/transformer_block_8/attention/key/add:z:0-model_8/transformer_block_8/attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
5model_8/transformer_block_8/attention/softmax/SoftmaxSoftmax<model_8/transformer_block_8/attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
6model_8/transformer_block_8/attention/dropout/IdentityIdentity?model_8/transformer_block_8/attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������		�
5model_8/transformer_block_8/attention/einsum_1/EinsumEinsum?model_8/transformer_block_8/attention/dropout/Identity:output:03model_8/transformer_block_8/attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
Smodel_8/transformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp\model_8_transformer_block_8_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Dmodel_8/transformer_block_8/attention/attention_output/einsum/EinsumEinsum>model_8/transformer_block_8/attention/einsum_1/Einsum:output:0[model_8/transformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
Imodel_8/transformer_block_8/attention/attention_output/add/ReadVariableOpReadVariableOpRmodel_8_transformer_block_8_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
:model_8/transformer_block_8/attention/attention_output/addAddV2Mmodel_8/transformer_block_8/attention/attention_output/einsum/Einsum:output:0Qmodel_8/transformer_block_8/attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
/model_8/transformer_block_8/dropout_48/IdentityIdentity>model_8/transformer_block_8/attention/attention_output/add:z:0*
T0*+
_output_shapes
:���������	�
model_8/transformer_block_8/addAddV2.model_8/token_and_position_embedding_8/add:z:08model_8/transformer_block_8/dropout_48/Identity:output:0*
T0*+
_output_shapes
:���������	�
Qmodel_8/transformer_block_8/layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
?model_8/transformer_block_8/layer_normalization_16/moments/meanMean#model_8/transformer_block_8/add:z:0Zmodel_8/transformer_block_8/layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
Gmodel_8/transformer_block_8/layer_normalization_16/moments/StopGradientStopGradientHmodel_8/transformer_block_8/layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Lmodel_8/transformer_block_8/layer_normalization_16/moments/SquaredDifferenceSquaredDifference#model_8/transformer_block_8/add:z:0Pmodel_8/transformer_block_8/layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Umodel_8/transformer_block_8/layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Cmodel_8/transformer_block_8/layer_normalization_16/moments/varianceMeanPmodel_8/transformer_block_8/layer_normalization_16/moments/SquaredDifference:z:0^model_8/transformer_block_8/layer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
@model_8/transformer_block_8/layer_normalization_16/batchnorm/addAddV2Lmodel_8/transformer_block_8/layer_normalization_16/moments/variance:output:0Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/RsqrtRsqrtDmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Omodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_8_transformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
@model_8/transformer_block_8/layer_normalization_16/batchnorm/mulMulFmodel_8/transformer_block_8/layer_normalization_16/batchnorm/Rsqrt:y:0Wmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_1Mul#model_8/transformer_block_8/add:z:0Dmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_2MulHmodel_8/transformer_block_8/layer_normalization_16/moments/mean:output:0Dmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOpTmodel_8_transformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
@model_8/transformer_block_8/layer_normalization_16/batchnorm/subSubSmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp:value:0Fmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_1:z:0Dmodel_8/transformer_block_8/layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
Jmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpReadVariableOpSmodel_8_transformer_block_8_sequential_8_dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
@model_8/transformer_block_8/sequential_8/dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
@model_8/transformer_block_8/sequential_8/dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Amodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/ShapeShapeFmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Imodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2GatherV2Jmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Shape:output:0Imodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/free:output:0Rmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Fmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1GatherV2Jmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Shape:output:0Imodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/axes:output:0Tmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Amodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
@model_8/transformer_block_8/sequential_8/dense_56/Tensordot/ProdProdMmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2:output:0Jmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Cmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Bmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Prod_1ProdOmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1:output:0Lmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Gmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Bmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/concatConcatV2Imodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/free:output:0Imodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/axes:output:0Pmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Amodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/stackPackImodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Prod:output:0Kmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Emodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/transpose	TransposeFmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0Kmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Cmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/ReshapeReshapeImodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/transpose:y:0Jmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Bmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/MatMulMatMulLmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Reshape:output:0Rmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Cmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Imodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/concat_1ConcatV2Mmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2:output:0Lmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/Const_2:output:0Rmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
;model_8/transformer_block_8/sequential_8/dense_56/TensordotReshapeLmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/MatMul:product:0Mmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
Hmodel_8/transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOpReadVariableOpQmodel_8_transformer_block_8_sequential_8_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
9model_8/transformer_block_8/sequential_8/dense_56/BiasAddBiasAddDmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot:output:0Pmodel_8/transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
6model_8/transformer_block_8/sequential_8/dense_56/ReluReluBmodel_8/transformer_block_8/sequential_8/dense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
Jmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOpReadVariableOpSmodel_8_transformer_block_8_sequential_8_dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
@model_8/transformer_block_8/sequential_8/dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
@model_8/transformer_block_8/sequential_8/dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Amodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/ShapeShapeDmodel_8/transformer_block_8/sequential_8/dense_56/Relu:activations:0*
T0*
_output_shapes
:�
Imodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2GatherV2Jmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Shape:output:0Imodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/free:output:0Rmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Fmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1GatherV2Jmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Shape:output:0Imodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/axes:output:0Tmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Amodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
@model_8/transformer_block_8/sequential_8/dense_57/Tensordot/ProdProdMmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2:output:0Jmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Cmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Bmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Prod_1ProdOmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1:output:0Lmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Gmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Bmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/concatConcatV2Imodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/free:output:0Imodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/axes:output:0Pmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Amodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/stackPackImodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Prod:output:0Kmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Emodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/transpose	TransposeDmodel_8/transformer_block_8/sequential_8/dense_56/Relu:activations:0Kmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Cmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/ReshapeReshapeImodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/transpose:y:0Jmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Bmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/MatMulMatMulLmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Reshape:output:0Rmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Cmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Imodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/concat_1ConcatV2Mmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2:output:0Lmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/Const_2:output:0Rmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
;model_8/transformer_block_8/sequential_8/dense_57/TensordotReshapeLmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/MatMul:product:0Mmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
Hmodel_8/transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOpReadVariableOpQmodel_8_transformer_block_8_sequential_8_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
9model_8/transformer_block_8/sequential_8/dense_57/BiasAddBiasAddDmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot:output:0Pmodel_8/transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
/model_8/transformer_block_8/dropout_49/IdentityIdentityBmodel_8/transformer_block_8/sequential_8/dense_57/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
!model_8/transformer_block_8/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1:z:08model_8/transformer_block_8/dropout_49/Identity:output:0*
T0*+
_output_shapes
:���������	�
Qmodel_8/transformer_block_8/layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
?model_8/transformer_block_8/layer_normalization_17/moments/meanMean%model_8/transformer_block_8/add_1:z:0Zmodel_8/transformer_block_8/layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
Gmodel_8/transformer_block_8/layer_normalization_17/moments/StopGradientStopGradientHmodel_8/transformer_block_8/layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Lmodel_8/transformer_block_8/layer_normalization_17/moments/SquaredDifferenceSquaredDifference%model_8/transformer_block_8/add_1:z:0Pmodel_8/transformer_block_8/layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Umodel_8/transformer_block_8/layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Cmodel_8/transformer_block_8/layer_normalization_17/moments/varianceMeanPmodel_8/transformer_block_8/layer_normalization_17/moments/SquaredDifference:z:0^model_8/transformer_block_8/layer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
@model_8/transformer_block_8/layer_normalization_17/batchnorm/addAddV2Lmodel_8/transformer_block_8/layer_normalization_17/moments/variance:output:0Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/RsqrtRsqrtDmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Omodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_8_transformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
@model_8/transformer_block_8/layer_normalization_17/batchnorm/mulMulFmodel_8/transformer_block_8/layer_normalization_17/batchnorm/Rsqrt:y:0Wmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_1Mul%model_8/transformer_block_8/add_1:z:0Dmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_2MulHmodel_8/transformer_block_8/layer_normalization_17/moments/mean:output:0Dmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOpTmodel_8_transformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
@model_8/transformer_block_8/layer_normalization_17/batchnorm/subSubSmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp:value:0Fmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_1:z:0Dmodel_8/transformer_block_8/layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
model_8/dropout_52/IdentityIdentity2model_8/batch_normalization_34/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� {
9model_8/global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'model_8/global_average_pooling1d_8/MeanMeanFmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add_1:z:0Bmodel_8/global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
&model_8/dense_61/MatMul/ReadVariableOpReadVariableOp/model_8_dense_61_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_8/dense_61/MatMulMatMul$model_8/dropout_52/Identity:output:0.model_8/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_8/dense_61/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_8/dense_61/BiasAddBiasAdd!model_8/dense_61/MatMul:product:0/model_8/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
model_8/dense_61/ReluRelu!model_8/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
&model_8/dense_58/MatMul/ReadVariableOpReadVariableOp/model_8_dense_58_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
model_8/dense_58/MatMulMatMul0model_8/global_average_pooling1d_8/Mean:output:0.model_8/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'model_8/dense_58/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_8/dense_58/BiasAddBiasAdd!model_8/dense_58/MatMul:product:0/model_8/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
model_8/dense_58/ReluRelu!model_8/dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
7model_8/batch_normalization_35/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_35_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0s
.model_8/batch_normalization_35/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,model_8/batch_normalization_35/batchnorm/addAddV2?model_8/batch_normalization_35/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_35/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_35/batchnorm/RsqrtRsqrt0model_8/batch_normalization_35/batchnorm/add:z:0*
T0*
_output_shapes
: �
;model_8/batch_normalization_35/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_35_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
,model_8/batch_normalization_35/batchnorm/mulMul2model_8/batch_normalization_35/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_35/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_35/batchnorm/mul_1Mul#model_8/dense_61/Relu:activations:00model_8/batch_normalization_35/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
9model_8/batch_normalization_35/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_35_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
.model_8/batch_normalization_35/batchnorm/mul_2MulAmodel_8/batch_normalization_35/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_35/batchnorm/mul:z:0*
T0*
_output_shapes
: �
9model_8/batch_normalization_35/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_35_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
,model_8/batch_normalization_35/batchnorm/subSubAmodel_8/batch_normalization_35/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_35/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_35/batchnorm/add_1AddV22model_8/batch_normalization_35/batchnorm/mul_1:z:00model_8/batch_normalization_35/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
7model_8/batch_normalization_32/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0s
.model_8/batch_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,model_8/batch_normalization_32/batchnorm/addAddV2?model_8/batch_normalization_32/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_32/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_32/batchnorm/RsqrtRsqrt0model_8/batch_normalization_32/batchnorm/add:z:0*
T0*
_output_shapes
: �
;model_8/batch_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
,model_8/batch_normalization_32/batchnorm/mulMul2model_8/batch_normalization_32/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_32/batchnorm/mul_1Mul#model_8/dense_58/Relu:activations:00model_8/batch_normalization_32/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
9model_8/batch_normalization_32/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_32_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
.model_8/batch_normalization_32/batchnorm/mul_2MulAmodel_8/batch_normalization_32/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_32/batchnorm/mul:z:0*
T0*
_output_shapes
: �
9model_8/batch_normalization_32/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_32_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
,model_8/batch_normalization_32/batchnorm/subSubAmodel_8/batch_normalization_32/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_32/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
.model_8/batch_normalization_32/batchnorm/add_1AddV22model_8/batch_normalization_32/batchnorm/mul_1:z:00model_8/batch_normalization_32/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
model_8/dropout_50/IdentityIdentity2model_8/batch_normalization_32/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� �
model_8/dropout_53/IdentityIdentity2model_8/batch_normalization_35/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� c
!model_8/concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_8/concatenate_8/concatConcatV2$model_8/dropout_50/Identity:output:0$model_8/dropout_53/Identity:output:0*model_8/concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
&model_8/dense_62/MatMul/ReadVariableOpReadVariableOp/model_8_dense_62_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_8/dense_62/MatMulMatMul%model_8/concatenate_8/concat:output:0.model_8/dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_8/dense_62/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_8/dense_62/BiasAddBiasAdd!model_8/dense_62/MatMul:product:0/model_8/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
model_8/dense_62/SoftmaxSoftmax!model_8/dense_62/BiasAdd:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"model_8/dense_62/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp8^model_8/batch_normalization_32/batchnorm/ReadVariableOp:^model_8/batch_normalization_32/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_32/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_32/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_33/batchnorm/ReadVariableOp:^model_8/batch_normalization_33/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_33/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_33/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_34/batchnorm/ReadVariableOp:^model_8/batch_normalization_34/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_34/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_34/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_35/batchnorm/ReadVariableOp:^model_8/batch_normalization_35/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_35/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_35/batchnorm/mul/ReadVariableOp(^model_8/dense_58/BiasAdd/ReadVariableOp'^model_8/dense_58/MatMul/ReadVariableOp(^model_8/dense_59/BiasAdd/ReadVariableOp'^model_8/dense_59/MatMul/ReadVariableOp(^model_8/dense_60/BiasAdd/ReadVariableOp'^model_8/dense_60/MatMul/ReadVariableOp(^model_8/dense_61/BiasAdd/ReadVariableOp'^model_8/dense_61/MatMul/ReadVariableOp(^model_8/dense_62/BiasAdd/ReadVariableOp'^model_8/dense_62/MatMul/ReadVariableOpE^model_8/token_and_position_embedding_8/embedding_16/embedding_lookupE^model_8/token_and_position_embedding_8/embedding_17/embedding_lookupJ^model_8/transformer_block_8/attention/attention_output/add/ReadVariableOpT^model_8/transformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp=^model_8/transformer_block_8/attention/key/add/ReadVariableOpG^model_8/transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp?^model_8/transformer_block_8/attention/query/add/ReadVariableOpI^model_8/transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp?^model_8/transformer_block_8/attention/value/add/ReadVariableOpI^model_8/transformer_block_8/attention/value/einsum/Einsum/ReadVariableOpL^model_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpP^model_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpL^model_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpP^model_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpI^model_8/transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOpK^model_8/transformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpI^model_8/transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOpK^model_8/transformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7model_8/batch_normalization_32/batchnorm/ReadVariableOp7model_8/batch_normalization_32/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_32/batchnorm/ReadVariableOp_19model_8/batch_normalization_32/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_32/batchnorm/ReadVariableOp_29model_8/batch_normalization_32/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_32/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_32/batchnorm/mul/ReadVariableOp2r
7model_8/batch_normalization_33/batchnorm/ReadVariableOp7model_8/batch_normalization_33/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_33/batchnorm/ReadVariableOp_19model_8/batch_normalization_33/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_33/batchnorm/ReadVariableOp_29model_8/batch_normalization_33/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_33/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_33/batchnorm/mul/ReadVariableOp2r
7model_8/batch_normalization_34/batchnorm/ReadVariableOp7model_8/batch_normalization_34/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_34/batchnorm/ReadVariableOp_19model_8/batch_normalization_34/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_34/batchnorm/ReadVariableOp_29model_8/batch_normalization_34/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_34/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_34/batchnorm/mul/ReadVariableOp2r
7model_8/batch_normalization_35/batchnorm/ReadVariableOp7model_8/batch_normalization_35/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_35/batchnorm/ReadVariableOp_19model_8/batch_normalization_35/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_35/batchnorm/ReadVariableOp_29model_8/batch_normalization_35/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_35/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_35/batchnorm/mul/ReadVariableOp2R
'model_8/dense_58/BiasAdd/ReadVariableOp'model_8/dense_58/BiasAdd/ReadVariableOp2P
&model_8/dense_58/MatMul/ReadVariableOp&model_8/dense_58/MatMul/ReadVariableOp2R
'model_8/dense_59/BiasAdd/ReadVariableOp'model_8/dense_59/BiasAdd/ReadVariableOp2P
&model_8/dense_59/MatMul/ReadVariableOp&model_8/dense_59/MatMul/ReadVariableOp2R
'model_8/dense_60/BiasAdd/ReadVariableOp'model_8/dense_60/BiasAdd/ReadVariableOp2P
&model_8/dense_60/MatMul/ReadVariableOp&model_8/dense_60/MatMul/ReadVariableOp2R
'model_8/dense_61/BiasAdd/ReadVariableOp'model_8/dense_61/BiasAdd/ReadVariableOp2P
&model_8/dense_61/MatMul/ReadVariableOp&model_8/dense_61/MatMul/ReadVariableOp2R
'model_8/dense_62/BiasAdd/ReadVariableOp'model_8/dense_62/BiasAdd/ReadVariableOp2P
&model_8/dense_62/MatMul/ReadVariableOp&model_8/dense_62/MatMul/ReadVariableOp2�
Dmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookupDmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup2�
Dmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookupDmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup2�
Imodel_8/transformer_block_8/attention/attention_output/add/ReadVariableOpImodel_8/transformer_block_8/attention/attention_output/add/ReadVariableOp2�
Smodel_8/transformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOpSmodel_8/transformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp2|
<model_8/transformer_block_8/attention/key/add/ReadVariableOp<model_8/transformer_block_8/attention/key/add/ReadVariableOp2�
Fmodel_8/transformer_block_8/attention/key/einsum/Einsum/ReadVariableOpFmodel_8/transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp2�
>model_8/transformer_block_8/attention/query/add/ReadVariableOp>model_8/transformer_block_8/attention/query/add/ReadVariableOp2�
Hmodel_8/transformer_block_8/attention/query/einsum/Einsum/ReadVariableOpHmodel_8/transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp2�
>model_8/transformer_block_8/attention/value/add/ReadVariableOp>model_8/transformer_block_8/attention/value/add/ReadVariableOp2�
Hmodel_8/transformer_block_8/attention/value/einsum/Einsum/ReadVariableOpHmodel_8/transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp2�
Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpKmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp2�
Omodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpOmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp2�
Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpKmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp2�
Omodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpOmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp2�
Hmodel_8/transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOpHmodel_8/transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp2�
Jmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpJmodel_8/transformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp2�
Hmodel_8/transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOpHmodel_8/transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp2�
Jmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOpJmodel_8/transformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_17:QM
'
_output_shapes
:���������

"
_user_specified_name
input_18
�
�
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9511332

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

�
E__inference_dense_62_layer_call_and_return_conditional_losses_9508623

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
��
�?
 __inference__traced_save_9512145
file_prefix.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop;
7savev2_batch_normalization_33_gamma_read_readvariableop:
6savev2_batch_normalization_33_beta_read_readvariableopA
=savev2_batch_normalization_33_moving_mean_read_readvariableopE
Asavev2_batch_normalization_33_moving_variance_read_readvariableop.
*savev2_dense_60_kernel_read_readvariableop,
(savev2_dense_60_bias_read_readvariableop;
7savev2_batch_normalization_34_gamma_read_readvariableop:
6savev2_batch_normalization_34_beta_read_readvariableopA
=savev2_batch_normalization_34_moving_mean_read_readvariableopE
Asavev2_batch_normalization_34_moving_variance_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop;
7savev2_batch_normalization_32_gamma_read_readvariableop:
6savev2_batch_normalization_32_beta_read_readvariableopA
=savev2_batch_normalization_32_moving_mean_read_readvariableopE
Asavev2_batch_normalization_32_moving_variance_read_readvariableop;
7savev2_batch_normalization_35_gamma_read_readvariableop:
6savev2_batch_normalization_35_beta_read_readvariableopA
=savev2_batch_normalization_35_moving_mean_read_readvariableopE
Asavev2_batch_normalization_35_moving_variance_read_readvariableop.
*savev2_dense_62_kernel_read_readvariableop,
(savev2_dense_62_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	U
Qsavev2_token_and_position_embedding_8_embedding_16_embeddings_read_readvariableopU
Qsavev2_token_and_position_embedding_8_embedding_17_embeddings_read_readvariableopI
Esavev2_transformer_block_8_attention_query_kernel_read_readvariableopG
Csavev2_transformer_block_8_attention_query_bias_read_readvariableopG
Csavev2_transformer_block_8_attention_key_kernel_read_readvariableopE
Asavev2_transformer_block_8_attention_key_bias_read_readvariableopI
Esavev2_transformer_block_8_attention_value_kernel_read_readvariableopG
Csavev2_transformer_block_8_attention_value_bias_read_readvariableopT
Psavev2_transformer_block_8_attention_attention_output_kernel_read_readvariableopR
Nsavev2_transformer_block_8_attention_attention_output_bias_read_readvariableop.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableopO
Ksavev2_transformer_block_8_layer_normalization_16_gamma_read_readvariableopN
Jsavev2_transformer_block_8_layer_normalization_16_beta_read_readvariableopO
Ksavev2_transformer_block_8_layer_normalization_17_gamma_read_readvariableopN
Jsavev2_transformer_block_8_layer_normalization_17_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_59_kernel_m_read_readvariableop3
/savev2_adam_dense_59_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_m_read_readvariableop5
1savev2_adam_dense_60_kernel_m_read_readvariableop3
/savev2_adam_dense_60_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_m_read_readvariableop5
1savev2_adam_dense_58_kernel_m_read_readvariableop3
/savev2_adam_dense_58_bias_m_read_readvariableop5
1savev2_adam_dense_61_kernel_m_read_readvariableop3
/savev2_adam_dense_61_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_m_read_readvariableopB
>savev2_adam_batch_normalization_35_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_35_beta_m_read_readvariableop5
1savev2_adam_dense_62_kernel_m_read_readvariableop3
/savev2_adam_dense_62_bias_m_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_8_embedding_16_embeddings_m_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_8_embedding_17_embeddings_m_read_readvariableopP
Lsavev2_adam_transformer_block_8_attention_query_kernel_m_read_readvariableopN
Jsavev2_adam_transformer_block_8_attention_query_bias_m_read_readvariableopN
Jsavev2_adam_transformer_block_8_attention_key_kernel_m_read_readvariableopL
Hsavev2_adam_transformer_block_8_attention_key_bias_m_read_readvariableopP
Lsavev2_adam_transformer_block_8_attention_value_kernel_m_read_readvariableopN
Jsavev2_adam_transformer_block_8_attention_value_bias_m_read_readvariableop[
Wsavev2_adam_transformer_block_8_attention_attention_output_kernel_m_read_readvariableopY
Usavev2_adam_transformer_block_8_attention_attention_output_bias_m_read_readvariableop5
1savev2_adam_dense_56_kernel_m_read_readvariableop3
/savev2_adam_dense_56_bias_m_read_readvariableop5
1savev2_adam_dense_57_kernel_m_read_readvariableop3
/savev2_adam_dense_57_bias_m_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_16_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_16_beta_m_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_17_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_17_beta_m_read_readvariableop5
1savev2_adam_dense_59_kernel_v_read_readvariableop3
/savev2_adam_dense_59_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_v_read_readvariableop5
1savev2_adam_dense_60_kernel_v_read_readvariableop3
/savev2_adam_dense_60_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_v_read_readvariableop5
1savev2_adam_dense_58_kernel_v_read_readvariableop3
/savev2_adam_dense_58_bias_v_read_readvariableop5
1savev2_adam_dense_61_kernel_v_read_readvariableop3
/savev2_adam_dense_61_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_v_read_readvariableopB
>savev2_adam_batch_normalization_35_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_35_beta_v_read_readvariableop5
1savev2_adam_dense_62_kernel_v_read_readvariableop3
/savev2_adam_dense_62_bias_v_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_8_embedding_16_embeddings_v_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_8_embedding_17_embeddings_v_read_readvariableopP
Lsavev2_adam_transformer_block_8_attention_query_kernel_v_read_readvariableopN
Jsavev2_adam_transformer_block_8_attention_query_bias_v_read_readvariableopN
Jsavev2_adam_transformer_block_8_attention_key_kernel_v_read_readvariableopL
Hsavev2_adam_transformer_block_8_attention_key_bias_v_read_readvariableopP
Lsavev2_adam_transformer_block_8_attention_value_kernel_v_read_readvariableopN
Jsavev2_adam_transformer_block_8_attention_value_bias_v_read_readvariableop[
Wsavev2_adam_transformer_block_8_attention_attention_output_kernel_v_read_readvariableopY
Usavev2_adam_transformer_block_8_attention_attention_output_bias_v_read_readvariableop5
1savev2_adam_dense_56_kernel_v_read_readvariableop3
/savev2_adam_dense_56_bias_v_read_readvariableop5
1savev2_adam_dense_57_kernel_v_read_readvariableop3
/savev2_adam_dense_57_bias_v_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_16_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_16_beta_v_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_17_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_17_beta_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop7savev2_batch_normalization_33_gamma_read_readvariableop6savev2_batch_normalization_33_beta_read_readvariableop=savev2_batch_normalization_33_moving_mean_read_readvariableopAsavev2_batch_normalization_33_moving_variance_read_readvariableop*savev2_dense_60_kernel_read_readvariableop(savev2_dense_60_bias_read_readvariableop7savev2_batch_normalization_34_gamma_read_readvariableop6savev2_batch_normalization_34_beta_read_readvariableop=savev2_batch_normalization_34_moving_mean_read_readvariableopAsavev2_batch_normalization_34_moving_variance_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop7savev2_batch_normalization_32_gamma_read_readvariableop6savev2_batch_normalization_32_beta_read_readvariableop=savev2_batch_normalization_32_moving_mean_read_readvariableopAsavev2_batch_normalization_32_moving_variance_read_readvariableop7savev2_batch_normalization_35_gamma_read_readvariableop6savev2_batch_normalization_35_beta_read_readvariableop=savev2_batch_normalization_35_moving_mean_read_readvariableopAsavev2_batch_normalization_35_moving_variance_read_readvariableop*savev2_dense_62_kernel_read_readvariableop(savev2_dense_62_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableopQsavev2_token_and_position_embedding_8_embedding_16_embeddings_read_readvariableopQsavev2_token_and_position_embedding_8_embedding_17_embeddings_read_readvariableopEsavev2_transformer_block_8_attention_query_kernel_read_readvariableopCsavev2_transformer_block_8_attention_query_bias_read_readvariableopCsavev2_transformer_block_8_attention_key_kernel_read_readvariableopAsavev2_transformer_block_8_attention_key_bias_read_readvariableopEsavev2_transformer_block_8_attention_value_kernel_read_readvariableopCsavev2_transformer_block_8_attention_value_bias_read_readvariableopPsavev2_transformer_block_8_attention_attention_output_kernel_read_readvariableopNsavev2_transformer_block_8_attention_attention_output_bias_read_readvariableop*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableopKsavev2_transformer_block_8_layer_normalization_16_gamma_read_readvariableopJsavev2_transformer_block_8_layer_normalization_16_beta_read_readvariableopKsavev2_transformer_block_8_layer_normalization_17_gamma_read_readvariableopJsavev2_transformer_block_8_layer_normalization_17_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_59_kernel_m_read_readvariableop/savev2_adam_dense_59_bias_m_read_readvariableop>savev2_adam_batch_normalization_33_gamma_m_read_readvariableop=savev2_adam_batch_normalization_33_beta_m_read_readvariableop1savev2_adam_dense_60_kernel_m_read_readvariableop/savev2_adam_dense_60_bias_m_read_readvariableop>savev2_adam_batch_normalization_34_gamma_m_read_readvariableop=savev2_adam_batch_normalization_34_beta_m_read_readvariableop1savev2_adam_dense_58_kernel_m_read_readvariableop/savev2_adam_dense_58_bias_m_read_readvariableop1savev2_adam_dense_61_kernel_m_read_readvariableop/savev2_adam_dense_61_bias_m_read_readvariableop>savev2_adam_batch_normalization_32_gamma_m_read_readvariableop=savev2_adam_batch_normalization_32_beta_m_read_readvariableop>savev2_adam_batch_normalization_35_gamma_m_read_readvariableop=savev2_adam_batch_normalization_35_beta_m_read_readvariableop1savev2_adam_dense_62_kernel_m_read_readvariableop/savev2_adam_dense_62_bias_m_read_readvariableopXsavev2_adam_token_and_position_embedding_8_embedding_16_embeddings_m_read_readvariableopXsavev2_adam_token_and_position_embedding_8_embedding_17_embeddings_m_read_readvariableopLsavev2_adam_transformer_block_8_attention_query_kernel_m_read_readvariableopJsavev2_adam_transformer_block_8_attention_query_bias_m_read_readvariableopJsavev2_adam_transformer_block_8_attention_key_kernel_m_read_readvariableopHsavev2_adam_transformer_block_8_attention_key_bias_m_read_readvariableopLsavev2_adam_transformer_block_8_attention_value_kernel_m_read_readvariableopJsavev2_adam_transformer_block_8_attention_value_bias_m_read_readvariableopWsavev2_adam_transformer_block_8_attention_attention_output_kernel_m_read_readvariableopUsavev2_adam_transformer_block_8_attention_attention_output_bias_m_read_readvariableop1savev2_adam_dense_56_kernel_m_read_readvariableop/savev2_adam_dense_56_bias_m_read_readvariableop1savev2_adam_dense_57_kernel_m_read_readvariableop/savev2_adam_dense_57_bias_m_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_16_gamma_m_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_16_beta_m_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_17_gamma_m_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_17_beta_m_read_readvariableop1savev2_adam_dense_59_kernel_v_read_readvariableop/savev2_adam_dense_59_bias_v_read_readvariableop>savev2_adam_batch_normalization_33_gamma_v_read_readvariableop=savev2_adam_batch_normalization_33_beta_v_read_readvariableop1savev2_adam_dense_60_kernel_v_read_readvariableop/savev2_adam_dense_60_bias_v_read_readvariableop>savev2_adam_batch_normalization_34_gamma_v_read_readvariableop=savev2_adam_batch_normalization_34_beta_v_read_readvariableop1savev2_adam_dense_58_kernel_v_read_readvariableop/savev2_adam_dense_58_bias_v_read_readvariableop1savev2_adam_dense_61_kernel_v_read_readvariableop/savev2_adam_dense_61_bias_v_read_readvariableop>savev2_adam_batch_normalization_32_gamma_v_read_readvariableop=savev2_adam_batch_normalization_32_beta_v_read_readvariableop>savev2_adam_batch_normalization_35_gamma_v_read_readvariableop=savev2_adam_batch_normalization_35_beta_v_read_readvariableop1savev2_adam_dense_62_kernel_v_read_readvariableop/savev2_adam_dense_62_bias_v_read_readvariableopXsavev2_adam_token_and_position_embedding_8_embedding_16_embeddings_v_read_readvariableopXsavev2_adam_token_and_position_embedding_8_embedding_17_embeddings_v_read_readvariableopLsavev2_adam_transformer_block_8_attention_query_kernel_v_read_readvariableopJsavev2_adam_transformer_block_8_attention_query_bias_v_read_readvariableopJsavev2_adam_transformer_block_8_attention_key_kernel_v_read_readvariableopHsavev2_adam_transformer_block_8_attention_key_bias_v_read_readvariableopLsavev2_adam_transformer_block_8_attention_value_kernel_v_read_readvariableopJsavev2_adam_transformer_block_8_attention_value_bias_v_read_readvariableopWsavev2_adam_transformer_block_8_attention_attention_output_kernel_v_read_readvariableopUsavev2_adam_transformer_block_8_attention_attention_output_bias_v_read_readvariableop1savev2_adam_dense_56_kernel_v_read_readvariableop/savev2_adam_dense_56_bias_v_read_readvariableop1savev2_adam_dense_57_kernel_v_read_readvariableop/savev2_adam_dense_57_bias_v_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_16_gamma_v_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_16_beta_v_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_17_gamma_v_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_17_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
8__inference_batch_normalization_34_layer_call_fn_9511154

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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9508081o
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
�
H
,__inference_dropout_53_layer_call_fn_9511478

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
G__inference_dropout_53_layer_call_and_return_conditional_losses_9508601`
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
�
�
8__inference_batch_normalization_35_layer_call_fn_9511379

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9508211o
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
,__inference_dropout_53_layer_call_fn_9511483

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
G__inference_dropout_53_layer_call_and_return_conditional_losses_9508758o
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
�
[
/__inference_concatenate_8_layer_call_fn_9511506
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9508610`
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
�
�
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9511412

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

�
E__inference_dense_60_layer_call_and_return_conditional_losses_9508322

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
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507958

inputs"
dense_56_9507947:
dense_56_9507949:"
dense_57_9507952:
dense_57_9507954:
identity�� dense_56/StatefulPartitionedCall� dense_57/StatefulPartitionedCall�
 dense_56/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_9507947dense_56_9507949*
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
E__inference_dense_56_layer_call_and_return_conditional_losses_9507855�
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_9507952dense_57_9507954*
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
E__inference_dense_57_layer_call_and_return_conditional_losses_9507891|
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
E__inference_dense_60_layer_call_and_return_conditional_losses_9510787

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
�
t
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9508610

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
�
�
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9508353
x7
%embedding_17_embedding_lookup_9508340:	7
%embedding_16_embedding_lookup_9508346:
identity��embedding_16/embedding_lookup�embedding_17/embedding_lookup6
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
embedding_17/embedding_lookupResourceGather%embedding_17_embedding_lookup_9508340range:output:0*
Tindices0*8
_class.
,*loc:@embedding_17/embedding_lookup/9508340*
_output_shapes

:	*
dtype0�
&embedding_17/embedding_lookup/IdentityIdentity&embedding_17/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_17/embedding_lookup/9508340*
_output_shapes

:	�
(embedding_17/embedding_lookup/Identity_1Identity/embedding_17/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	]
embedding_16/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:���������	�
embedding_16/embedding_lookupResourceGather%embedding_16_embedding_lookup_9508346embedding_16/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_16/embedding_lookup/9508346*+
_output_shapes
:���������	*
dtype0�
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_16/embedding_lookup/9508346*+
_output_shapes
:���������	�
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
addAddV21embedding_16/embedding_lookup/Identity_1:output:01embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp^embedding_16/embedding_lookup^embedding_17/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2>
embedding_17/embedding_lookupembedding_17/embedding_lookup:J F
'
_output_shapes
:���������	

_user_specified_namex
�
�
*__inference_dense_56_layer_call_fn_9511682

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
E__inference_dense_56_layer_call_and_return_conditional_losses_9507855s
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
�
e
G__inference_dropout_50_layer_call_and_return_conditional_losses_9511461

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
8__inference_batch_normalization_33_layer_call_fn_9510652

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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9507806o
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9508211

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
�
�
.__inference_sequential_8_layer_call_fn_9507982
dense_56_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_56_inputunknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8� *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507958s
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
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������	
(
_user_specified_namedense_56_input
�
�
E__inference_dense_57_layer_call_and_return_conditional_losses_9507891

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
�
e
G__inference_dropout_52_layer_call_and_return_conditional_losses_9511234

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
f
G__inference_dropout_51_layer_call_and_return_conditional_losses_9510733

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
E__inference_dense_62_layer_call_and_return_conditional_losses_9511533

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
�
�

)__inference_model_8_layer_call_fn_9509901
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
GPU 2J 8� *M
fHRF
D__inference_model_8_layer_call_and_return_conditional_losses_9509302o
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
�%
�
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9511366

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
�
�
E__inference_dense_57_layer_call_and_return_conditional_losses_9511752

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
�

�
E__inference_dense_61_layer_call_and_return_conditional_losses_9508548

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
ʲ
�
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9510988

inputsK
5attention_query_einsum_einsum_readvariableop_resource:=
+attention_query_add_readvariableop_resource:I
3attention_key_einsum_einsum_readvariableop_resource:;
)attention_key_add_readvariableop_resource:K
5attention_value_einsum_einsum_readvariableop_resource:=
+attention_value_add_readvariableop_resource:V
@attention_attention_output_einsum_einsum_readvariableop_resource:D
6attention_attention_output_add_readvariableop_resource:J
<layer_normalization_16_batchnorm_mul_readvariableop_resource:F
8layer_normalization_16_batchnorm_readvariableop_resource:I
7sequential_8_dense_56_tensordot_readvariableop_resource:C
5sequential_8_dense_56_biasadd_readvariableop_resource:I
7sequential_8_dense_57_tensordot_readvariableop_resource:C
5sequential_8_dense_57_biasadd_readvariableop_resource:J
<layer_normalization_17_batchnorm_mul_readvariableop_resource:F
8layer_normalization_17_batchnorm_readvariableop_resource:
identity��-attention/attention_output/add/ReadVariableOp�7attention/attention_output/einsum/Einsum/ReadVariableOp� attention/key/add/ReadVariableOp�*attention/key/einsum/Einsum/ReadVariableOp�"attention/query/add/ReadVariableOp�,attention/query/einsum/Einsum/ReadVariableOp�"attention/value/add/ReadVariableOp�,attention/value/einsum/Einsum/ReadVariableOp�/layer_normalization_16/batchnorm/ReadVariableOp�3layer_normalization_16/batchnorm/mul/ReadVariableOp�/layer_normalization_17/batchnorm/ReadVariableOp�3layer_normalization_17/batchnorm/mul/ReadVariableOp�,sequential_8/dense_56/BiasAdd/ReadVariableOp�.sequential_8/dense_56/Tensordot/ReadVariableOp�,sequential_8/dense_57/BiasAdd/ReadVariableOp�.sequential_8/dense_57/Tensordot/ReadVariableOp�
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
:���������	y
dropout_48/IdentityIdentity"attention/attention_output/add:z:0*
T0*+
_output_shapes
:���������	h
addAddV2inputsdropout_48/Identity:output:0*
T0*+
_output_shapes
:���������	
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_16/moments/meanMeanadd:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(k
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/mul_1Muladd:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
.sequential_8/dense_56/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_8/dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_8/dense_56/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_8/dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_56/Tensordot/GatherV2GatherV2.sequential_8/dense_56/Tensordot/Shape:output:0-sequential_8/dense_56/Tensordot/free:output:06sequential_8/dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_8/dense_56/Tensordot/GatherV2_1GatherV2.sequential_8/dense_56/Tensordot/Shape:output:0-sequential_8/dense_56/Tensordot/axes:output:08sequential_8/dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_8/dense_56/Tensordot/ProdProd1sequential_8/dense_56/Tensordot/GatherV2:output:0.sequential_8/dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_8/dense_56/Tensordot/Prod_1Prod3sequential_8/dense_56/Tensordot/GatherV2_1:output:00sequential_8/dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_8/dense_56/Tensordot/concatConcatV2-sequential_8/dense_56/Tensordot/free:output:0-sequential_8/dense_56/Tensordot/axes:output:04sequential_8/dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_8/dense_56/Tensordot/stackPack-sequential_8/dense_56/Tensordot/Prod:output:0/sequential_8/dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_8/dense_56/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_8/dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
'sequential_8/dense_56/Tensordot/ReshapeReshape-sequential_8/dense_56/Tensordot/transpose:y:0.sequential_8/dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_8/dense_56/Tensordot/MatMulMatMul0sequential_8/dense_56/Tensordot/Reshape:output:06sequential_8/dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
'sequential_8/dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_8/dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_56/Tensordot/concat_1ConcatV21sequential_8/dense_56/Tensordot/GatherV2:output:00sequential_8/dense_56/Tensordot/Const_2:output:06sequential_8/dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_8/dense_56/TensordotReshape0sequential_8/dense_56/Tensordot/MatMul:product:01sequential_8/dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
,sequential_8/dense_56/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_8/dense_56/BiasAddBiasAdd(sequential_8/dense_56/Tensordot:output:04sequential_8/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
sequential_8/dense_56/ReluRelu&sequential_8/dense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
.sequential_8/dense_57/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_8/dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_8/dense_57/Tensordot/ShapeShape(sequential_8/dense_56/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_8/dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_57/Tensordot/GatherV2GatherV2.sequential_8/dense_57/Tensordot/Shape:output:0-sequential_8/dense_57/Tensordot/free:output:06sequential_8/dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_8/dense_57/Tensordot/GatherV2_1GatherV2.sequential_8/dense_57/Tensordot/Shape:output:0-sequential_8/dense_57/Tensordot/axes:output:08sequential_8/dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_8/dense_57/Tensordot/ProdProd1sequential_8/dense_57/Tensordot/GatherV2:output:0.sequential_8/dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_8/dense_57/Tensordot/Prod_1Prod3sequential_8/dense_57/Tensordot/GatherV2_1:output:00sequential_8/dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_8/dense_57/Tensordot/concatConcatV2-sequential_8/dense_57/Tensordot/free:output:0-sequential_8/dense_57/Tensordot/axes:output:04sequential_8/dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_8/dense_57/Tensordot/stackPack-sequential_8/dense_57/Tensordot/Prod:output:0/sequential_8/dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_8/dense_57/Tensordot/transpose	Transpose(sequential_8/dense_56/Relu:activations:0/sequential_8/dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
'sequential_8/dense_57/Tensordot/ReshapeReshape-sequential_8/dense_57/Tensordot/transpose:y:0.sequential_8/dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_8/dense_57/Tensordot/MatMulMatMul0sequential_8/dense_57/Tensordot/Reshape:output:06sequential_8/dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
'sequential_8/dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_8/dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_57/Tensordot/concat_1ConcatV21sequential_8/dense_57/Tensordot/GatherV2:output:00sequential_8/dense_57/Tensordot/Const_2:output:06sequential_8/dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_8/dense_57/TensordotReshape0sequential_8/dense_57/Tensordot/MatMul:product:01sequential_8/dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
,sequential_8/dense_57/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_8/dense_57/BiasAddBiasAdd(sequential_8/dense_57/Tensordot:output:04sequential_8/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	}
dropout_49/IdentityIdentity&sequential_8/dense_57/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
add_1AddV2*layer_normalization_16/batchnorm/add_1:z:0dropout_49/Identity:output:0*
T0*+
_output_shapes
:���������	
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_17/moments/meanMean	add_1:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
0layer_normalization_17/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(k
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	}
IdentityIdentity*layer_normalization_17/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp-^sequential_8/dense_56/BiasAdd/ReadVariableOp/^sequential_8/dense_56/Tensordot/ReadVariableOp-^sequential_8/dense_57/BiasAdd/ReadVariableOp/^sequential_8/dense_57/Tensordot/ReadVariableOp*"
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
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2\
,sequential_8/dense_56/BiasAdd/ReadVariableOp,sequential_8/dense_56/BiasAdd/ReadVariableOp2`
.sequential_8/dense_56/Tensordot/ReadVariableOp.sequential_8/dense_56/Tensordot/ReadVariableOp2\
,sequential_8/dense_57/BiasAdd/ReadVariableOp,sequential_8/dense_57/BiasAdd/ReadVariableOp2`
.sequential_8/dense_57/Tensordot/ReadVariableOp.sequential_8/dense_57/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�`
�
D__inference_model_8_layer_call_and_return_conditional_losses_9509597
input_17
input_18"
dense_59_9509491:
 
dense_59_9509493: ,
batch_normalization_33_9509496: ,
batch_normalization_33_9509498: ,
batch_normalization_33_9509500: ,
batch_normalization_33_9509502: "
dense_60_9509506:  
dense_60_9509508: 8
&token_and_position_embedding_8_9509511:	8
&token_and_position_embedding_8_9509513:,
batch_normalization_34_9509516: ,
batch_normalization_34_9509518: ,
batch_normalization_34_9509520: ,
batch_normalization_34_9509522: 1
transformer_block_8_9509525:-
transformer_block_8_9509527:1
transformer_block_8_9509529:-
transformer_block_8_9509531:1
transformer_block_8_9509533:-
transformer_block_8_9509535:1
transformer_block_8_9509537:)
transformer_block_8_9509539:)
transformer_block_8_9509541:)
transformer_block_8_9509543:-
transformer_block_8_9509545:)
transformer_block_8_9509547:-
transformer_block_8_9509549:)
transformer_block_8_9509551:)
transformer_block_8_9509553:)
transformer_block_8_9509555:"
dense_61_9509560:  
dense_61_9509562: "
dense_58_9509565: 
dense_58_9509567: ,
batch_normalization_35_9509570: ,
batch_normalization_35_9509572: ,
batch_normalization_35_9509574: ,
batch_normalization_35_9509576: ,
batch_normalization_32_9509579: ,
batch_normalization_32_9509581: ,
batch_normalization_32_9509583: ,
batch_normalization_32_9509585: "
dense_62_9509591:@
dense_62_9509593:
identity��.batch_normalization_32/StatefulPartitionedCall�.batch_normalization_33/StatefulPartitionedCall�.batch_normalization_34/StatefulPartitionedCall�.batch_normalization_35/StatefulPartitionedCall� dense_58/StatefulPartitionedCall� dense_59/StatefulPartitionedCall� dense_60/StatefulPartitionedCall� dense_61/StatefulPartitionedCall� dense_62/StatefulPartitionedCall�6token_and_position_embedding_8/StatefulPartitionedCall�+transformer_block_8/StatefulPartitionedCall�
 dense_59/StatefulPartitionedCallStatefulPartitionedCallinput_18dense_59_9509491dense_59_9509493*
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
E__inference_dense_59_layer_call_and_return_conditional_losses_9508289�
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0batch_normalization_33_9509496batch_normalization_33_9509498batch_normalization_33_9509500batch_normalization_33_9509502*
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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9507759�
dropout_51/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_9508309�
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#dropout_51/PartitionedCall:output:0dense_60_9509506dense_60_9509508*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_9508322�
6token_and_position_embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_17&token_and_position_embedding_8_9509511&token_and_position_embedding_8_9509513*
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
GPU 2J 8� *d
f_R]
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9508353�
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_34_9509516batch_normalization_34_9509518batch_normalization_34_9509520batch_normalization_34_9509522*
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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9508034�
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_8/StatefulPartitionedCall:output:0transformer_block_8_9509525transformer_block_8_9509527transformer_block_8_9509529transformer_block_8_9509531transformer_block_8_9509533transformer_block_8_9509535transformer_block_8_9509537transformer_block_8_9509539transformer_block_8_9509541transformer_block_8_9509543transformer_block_8_9509545transformer_block_8_9509547transformer_block_8_9509549transformer_block_8_9509551transformer_block_8_9509553transformer_block_8_9509555*
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
GPU 2J 8� *Y
fTRR
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9508495�
dropout_52/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
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
G__inference_dropout_52_layer_call_and_return_conditional_losses_9508534�
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *`
f[RY
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9508102�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall#dropout_52/PartitionedCall:output:0dense_61_9509560dense_61_9509562*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_9508548�
 dense_58/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_58_9509565dense_58_9509567*
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
E__inference_dense_58_layer_call_and_return_conditional_losses_9508565�
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_35_9509570batch_normalization_35_9509572batch_normalization_35_9509574batch_normalization_35_9509576*
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9508211�
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_32_9509579batch_normalization_32_9509581batch_normalization_32_9509583batch_normalization_32_9509585*
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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9508129�
dropout_50/PartitionedCallPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_9508594�
dropout_53/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
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
G__inference_dropout_53_layer_call_and_return_conditional_losses_9508601�
concatenate_8/PartitionedCallPartitionedCall#dropout_50/PartitionedCall:output:0#dropout_53/PartitionedCall:output:0*
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9508610�
 dense_62/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_62_9509591dense_62_9509593*
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
GPU 2J 8� *N
fIRG
E__inference_dense_62_layer_call_and_return_conditional_losses_9508623x
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall7^token_and_position_embedding_8/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2p
6token_and_position_embedding_8/StatefulPartitionedCall6token_and_position_embedding_8/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_17:QM
'
_output_shapes
:���������

"
_user_specified_name
input_18
�%
�
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9511446

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
�
H
,__inference_dropout_50_layer_call_fn_9511451

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
G__inference_dropout_50_layer_call_and_return_conditional_losses_9508594`
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
�
�

)__inference_model_8_layer_call_fn_9508721
input_17
input_18
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
StatefulPartitionedCallStatefulPartitionedCallinput_17input_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *M
fHRF
D__inference_model_8_layer_call_and_return_conditional_losses_9508630o
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
input_17:QM
'
_output_shapes
:���������

"
_user_specified_name
input_18
�
�
5__inference_transformer_block_8_layer_call_fn_9510861

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
GPU 2J 8� *Y
fTRR
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9509007s
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
�
�
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9508034

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
�
5__inference_transformer_block_8_layer_call_fn_9510824

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
GPU 2J 8� *Y
fTRR
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9508495s
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
ʲ
�
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9508495

inputsK
5attention_query_einsum_einsum_readvariableop_resource:=
+attention_query_add_readvariableop_resource:I
3attention_key_einsum_einsum_readvariableop_resource:;
)attention_key_add_readvariableop_resource:K
5attention_value_einsum_einsum_readvariableop_resource:=
+attention_value_add_readvariableop_resource:V
@attention_attention_output_einsum_einsum_readvariableop_resource:D
6attention_attention_output_add_readvariableop_resource:J
<layer_normalization_16_batchnorm_mul_readvariableop_resource:F
8layer_normalization_16_batchnorm_readvariableop_resource:I
7sequential_8_dense_56_tensordot_readvariableop_resource:C
5sequential_8_dense_56_biasadd_readvariableop_resource:I
7sequential_8_dense_57_tensordot_readvariableop_resource:C
5sequential_8_dense_57_biasadd_readvariableop_resource:J
<layer_normalization_17_batchnorm_mul_readvariableop_resource:F
8layer_normalization_17_batchnorm_readvariableop_resource:
identity��-attention/attention_output/add/ReadVariableOp�7attention/attention_output/einsum/Einsum/ReadVariableOp� attention/key/add/ReadVariableOp�*attention/key/einsum/Einsum/ReadVariableOp�"attention/query/add/ReadVariableOp�,attention/query/einsum/Einsum/ReadVariableOp�"attention/value/add/ReadVariableOp�,attention/value/einsum/Einsum/ReadVariableOp�/layer_normalization_16/batchnorm/ReadVariableOp�3layer_normalization_16/batchnorm/mul/ReadVariableOp�/layer_normalization_17/batchnorm/ReadVariableOp�3layer_normalization_17/batchnorm/mul/ReadVariableOp�,sequential_8/dense_56/BiasAdd/ReadVariableOp�.sequential_8/dense_56/Tensordot/ReadVariableOp�,sequential_8/dense_57/BiasAdd/ReadVariableOp�.sequential_8/dense_57/Tensordot/ReadVariableOp�
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
:���������	y
dropout_48/IdentityIdentity"attention/attention_output/add:z:0*
T0*+
_output_shapes
:���������	h
addAddV2inputsdropout_48/Identity:output:0*
T0*+
_output_shapes
:���������	
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_16/moments/meanMeanadd:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(k
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/mul_1Muladd:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
.sequential_8/dense_56/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_8/dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_8/dense_56/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_8/dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_56/Tensordot/GatherV2GatherV2.sequential_8/dense_56/Tensordot/Shape:output:0-sequential_8/dense_56/Tensordot/free:output:06sequential_8/dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_8/dense_56/Tensordot/GatherV2_1GatherV2.sequential_8/dense_56/Tensordot/Shape:output:0-sequential_8/dense_56/Tensordot/axes:output:08sequential_8/dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_8/dense_56/Tensordot/ProdProd1sequential_8/dense_56/Tensordot/GatherV2:output:0.sequential_8/dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_8/dense_56/Tensordot/Prod_1Prod3sequential_8/dense_56/Tensordot/GatherV2_1:output:00sequential_8/dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_8/dense_56/Tensordot/concatConcatV2-sequential_8/dense_56/Tensordot/free:output:0-sequential_8/dense_56/Tensordot/axes:output:04sequential_8/dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_8/dense_56/Tensordot/stackPack-sequential_8/dense_56/Tensordot/Prod:output:0/sequential_8/dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_8/dense_56/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_8/dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
'sequential_8/dense_56/Tensordot/ReshapeReshape-sequential_8/dense_56/Tensordot/transpose:y:0.sequential_8/dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_8/dense_56/Tensordot/MatMulMatMul0sequential_8/dense_56/Tensordot/Reshape:output:06sequential_8/dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
'sequential_8/dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_8/dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_56/Tensordot/concat_1ConcatV21sequential_8/dense_56/Tensordot/GatherV2:output:00sequential_8/dense_56/Tensordot/Const_2:output:06sequential_8/dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_8/dense_56/TensordotReshape0sequential_8/dense_56/Tensordot/MatMul:product:01sequential_8/dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
,sequential_8/dense_56/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_8/dense_56/BiasAddBiasAdd(sequential_8/dense_56/Tensordot:output:04sequential_8/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
sequential_8/dense_56/ReluRelu&sequential_8/dense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
.sequential_8/dense_57/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_8/dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_8/dense_57/Tensordot/ShapeShape(sequential_8/dense_56/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_8/dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_57/Tensordot/GatherV2GatherV2.sequential_8/dense_57/Tensordot/Shape:output:0-sequential_8/dense_57/Tensordot/free:output:06sequential_8/dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_8/dense_57/Tensordot/GatherV2_1GatherV2.sequential_8/dense_57/Tensordot/Shape:output:0-sequential_8/dense_57/Tensordot/axes:output:08sequential_8/dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_8/dense_57/Tensordot/ProdProd1sequential_8/dense_57/Tensordot/GatherV2:output:0.sequential_8/dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_8/dense_57/Tensordot/Prod_1Prod3sequential_8/dense_57/Tensordot/GatherV2_1:output:00sequential_8/dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_8/dense_57/Tensordot/concatConcatV2-sequential_8/dense_57/Tensordot/free:output:0-sequential_8/dense_57/Tensordot/axes:output:04sequential_8/dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_8/dense_57/Tensordot/stackPack-sequential_8/dense_57/Tensordot/Prod:output:0/sequential_8/dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_8/dense_57/Tensordot/transpose	Transpose(sequential_8/dense_56/Relu:activations:0/sequential_8/dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
'sequential_8/dense_57/Tensordot/ReshapeReshape-sequential_8/dense_57/Tensordot/transpose:y:0.sequential_8/dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_8/dense_57/Tensordot/MatMulMatMul0sequential_8/dense_57/Tensordot/Reshape:output:06sequential_8/dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
'sequential_8/dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_8/dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_57/Tensordot/concat_1ConcatV21sequential_8/dense_57/Tensordot/GatherV2:output:00sequential_8/dense_57/Tensordot/Const_2:output:06sequential_8/dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_8/dense_57/TensordotReshape0sequential_8/dense_57/Tensordot/MatMul:product:01sequential_8/dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
,sequential_8/dense_57/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_8/dense_57/BiasAddBiasAdd(sequential_8/dense_57/Tensordot:output:04sequential_8/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	}
dropout_49/IdentityIdentity&sequential_8/dense_57/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
add_1AddV2*layer_normalization_16/batchnorm/add_1:z:0dropout_49/Identity:output:0*
T0*+
_output_shapes
:���������	
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_17/moments/meanMean	add_1:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
0layer_normalization_17/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(k
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	}
IdentityIdentity*layer_normalization_17/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp-^sequential_8/dense_56/BiasAdd/ReadVariableOp/^sequential_8/dense_56/Tensordot/ReadVariableOp-^sequential_8/dense_57/BiasAdd/ReadVariableOp/^sequential_8/dense_57/Tensordot/ReadVariableOp*"
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
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2\
,sequential_8/dense_56/BiasAdd/ReadVariableOp,sequential_8/dense_56/BiasAdd/ReadVariableOp2`
.sequential_8/dense_56/Tensordot/ReadVariableOp.sequential_8/dense_56/Tensordot/ReadVariableOp2\
,sequential_8/dense_57/BiasAdd/ReadVariableOp,sequential_8/dense_57/BiasAdd/ReadVariableOp2`
.sequential_8/dense_57/Tensordot/ReadVariableOp.sequential_8/dense_57/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9508129

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
e
G__inference_dropout_53_layer_call_and_return_conditional_losses_9508601

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
*__inference_dense_60_layer_call_fn_9510776

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
GPU 2J 8� *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_9508322o
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
*__inference_dense_62_layer_call_fn_9511522

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
GPU 2J 8� *N
fIRG
E__inference_dense_62_layer_call_and_return_conditional_losses_9508623o
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
�
s
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9508102

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
�?
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_9511673

inputs<
*dense_56_tensordot_readvariableop_resource:6
(dense_56_biasadd_readvariableop_resource:<
*dense_57_tensordot_readvariableop_resource:6
(dense_57_biasadd_readvariableop_resource:
identity��dense_56/BiasAdd/ReadVariableOp�!dense_56/Tensordot/ReadVariableOp�dense_57/BiasAdd/ReadVariableOp�!dense_57/Tensordot/ReadVariableOp�
!dense_56/Tensordot/ReadVariableOpReadVariableOp*dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_56/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_56/Tensordot/GatherV2GatherV2!dense_56/Tensordot/Shape:output:0 dense_56/Tensordot/free:output:0)dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_56/Tensordot/GatherV2_1GatherV2!dense_56/Tensordot/Shape:output:0 dense_56/Tensordot/axes:output:0+dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_56/Tensordot/ProdProd$dense_56/Tensordot/GatherV2:output:0!dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_56/Tensordot/Prod_1Prod&dense_56/Tensordot/GatherV2_1:output:0#dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_56/Tensordot/concatConcatV2 dense_56/Tensordot/free:output:0 dense_56/Tensordot/axes:output:0'dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_56/Tensordot/stackPack dense_56/Tensordot/Prod:output:0"dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_56/Tensordot/transpose	Transposeinputs"dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
dense_56/Tensordot/ReshapeReshape dense_56/Tensordot/transpose:y:0!dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_56/Tensordot/MatMulMatMul#dense_56/Tensordot/Reshape:output:0)dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_56/Tensordot/concat_1ConcatV2$dense_56/Tensordot/GatherV2:output:0#dense_56/Tensordot/Const_2:output:0)dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_56/TensordotReshape#dense_56/Tensordot/MatMul:product:0$dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_56/BiasAddBiasAdddense_56/Tensordot:output:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	f
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
!dense_57/Tensordot/ReadVariableOpReadVariableOp*dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_57/Tensordot/ShapeShapedense_56/Relu:activations:0*
T0*
_output_shapes
:b
 dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_57/Tensordot/GatherV2GatherV2!dense_57/Tensordot/Shape:output:0 dense_57/Tensordot/free:output:0)dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_57/Tensordot/GatherV2_1GatherV2!dense_57/Tensordot/Shape:output:0 dense_57/Tensordot/axes:output:0+dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_57/Tensordot/ProdProd$dense_57/Tensordot/GatherV2:output:0!dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_57/Tensordot/Prod_1Prod&dense_57/Tensordot/GatherV2_1:output:0#dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_57/Tensordot/concatConcatV2 dense_57/Tensordot/free:output:0 dense_57/Tensordot/axes:output:0'dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_57/Tensordot/stackPack dense_57/Tensordot/Prod:output:0"dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_57/Tensordot/transpose	Transposedense_56/Relu:activations:0"dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
dense_57/Tensordot/ReshapeReshape dense_57/Tensordot/transpose:y:0!dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_57/Tensordot/MatMulMatMul#dense_57/Tensordot/Reshape:output:0)dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_57/Tensordot/concat_1ConcatV2$dense_57/Tensordot/GatherV2:output:0#dense_57/Tensordot/Const_2:output:0)dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_57/TensordotReshape#dense_57/Tensordot/MatMul:product:0$dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_57/BiasAddBiasAdddense_57/Tensordot:output:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	l
IdentityIdentitydense_57/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp ^dense_56/BiasAdd/ReadVariableOp"^dense_56/Tensordot/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp"^dense_57/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2F
!dense_56/Tensordot/ReadVariableOp!dense_56/Tensordot/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2F
!dense_57/Tensordot/ReadVariableOp!dense_57/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
e
G__inference_dropout_53_layer_call_and_return_conditional_losses_9511488

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
͝
�.
D__inference_model_8_layer_call_and_return_conditional_losses_9510157
inputs_0
inputs_19
'dense_59_matmul_readvariableop_resource:
 6
(dense_59_biasadd_readvariableop_resource: F
8batch_normalization_33_batchnorm_readvariableop_resource: J
<batch_normalization_33_batchnorm_mul_readvariableop_resource: H
:batch_normalization_33_batchnorm_readvariableop_1_resource: H
:batch_normalization_33_batchnorm_readvariableop_2_resource: 9
'dense_60_matmul_readvariableop_resource:  6
(dense_60_biasadd_readvariableop_resource: V
Dtoken_and_position_embedding_8_embedding_17_embedding_lookup_9509945:	V
Dtoken_and_position_embedding_8_embedding_16_embedding_lookup_9509951:F
8batch_normalization_34_batchnorm_readvariableop_resource: J
<batch_normalization_34_batchnorm_mul_readvariableop_resource: H
:batch_normalization_34_batchnorm_readvariableop_1_resource: H
:batch_normalization_34_batchnorm_readvariableop_2_resource: _
Itransformer_block_8_attention_query_einsum_einsum_readvariableop_resource:Q
?transformer_block_8_attention_query_add_readvariableop_resource:]
Gtransformer_block_8_attention_key_einsum_einsum_readvariableop_resource:O
=transformer_block_8_attention_key_add_readvariableop_resource:_
Itransformer_block_8_attention_value_einsum_einsum_readvariableop_resource:Q
?transformer_block_8_attention_value_add_readvariableop_resource:j
Ttransformer_block_8_attention_attention_output_einsum_einsum_readvariableop_resource:X
Jtransformer_block_8_attention_attention_output_add_readvariableop_resource:^
Ptransformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource:Z
Ltransformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource:]
Ktransformer_block_8_sequential_8_dense_56_tensordot_readvariableop_resource:W
Itransformer_block_8_sequential_8_dense_56_biasadd_readvariableop_resource:]
Ktransformer_block_8_sequential_8_dense_57_tensordot_readvariableop_resource:W
Itransformer_block_8_sequential_8_dense_57_biasadd_readvariableop_resource:^
Ptransformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource:Z
Ltransformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource:9
'dense_61_matmul_readvariableop_resource:  6
(dense_61_biasadd_readvariableop_resource: 9
'dense_58_matmul_readvariableop_resource: 6
(dense_58_biasadd_readvariableop_resource: F
8batch_normalization_35_batchnorm_readvariableop_resource: J
<batch_normalization_35_batchnorm_mul_readvariableop_resource: H
:batch_normalization_35_batchnorm_readvariableop_1_resource: H
:batch_normalization_35_batchnorm_readvariableop_2_resource: F
8batch_normalization_32_batchnorm_readvariableop_resource: J
<batch_normalization_32_batchnorm_mul_readvariableop_resource: H
:batch_normalization_32_batchnorm_readvariableop_1_resource: H
:batch_normalization_32_batchnorm_readvariableop_2_resource: 9
'dense_62_matmul_readvariableop_resource:@6
(dense_62_biasadd_readvariableop_resource:
identity��/batch_normalization_32/batchnorm/ReadVariableOp�1batch_normalization_32/batchnorm/ReadVariableOp_1�1batch_normalization_32/batchnorm/ReadVariableOp_2�3batch_normalization_32/batchnorm/mul/ReadVariableOp�/batch_normalization_33/batchnorm/ReadVariableOp�1batch_normalization_33/batchnorm/ReadVariableOp_1�1batch_normalization_33/batchnorm/ReadVariableOp_2�3batch_normalization_33/batchnorm/mul/ReadVariableOp�/batch_normalization_34/batchnorm/ReadVariableOp�1batch_normalization_34/batchnorm/ReadVariableOp_1�1batch_normalization_34/batchnorm/ReadVariableOp_2�3batch_normalization_34/batchnorm/mul/ReadVariableOp�/batch_normalization_35/batchnorm/ReadVariableOp�1batch_normalization_35/batchnorm/ReadVariableOp_1�1batch_normalization_35/batchnorm/ReadVariableOp_2�3batch_normalization_35/batchnorm/mul/ReadVariableOp�dense_58/BiasAdd/ReadVariableOp�dense_58/MatMul/ReadVariableOp�dense_59/BiasAdd/ReadVariableOp�dense_59/MatMul/ReadVariableOp�dense_60/BiasAdd/ReadVariableOp�dense_60/MatMul/ReadVariableOp�dense_61/BiasAdd/ReadVariableOp�dense_61/MatMul/ReadVariableOp�dense_62/BiasAdd/ReadVariableOp�dense_62/MatMul/ReadVariableOp�<token_and_position_embedding_8/embedding_16/embedding_lookup�<token_and_position_embedding_8/embedding_17/embedding_lookup�Atransformer_block_8/attention/attention_output/add/ReadVariableOp�Ktransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp�4transformer_block_8/attention/key/add/ReadVariableOp�>transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp�6transformer_block_8/attention/query/add/ReadVariableOp�@transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp�6transformer_block_8/attention/value/add/ReadVariableOp�@transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp�Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp�Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp�Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp�Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp�@transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp�Btransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp�@transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp�Btransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp�
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype0}
dense_59/MatMulMatMulinputs_1&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_59/ReluReludense_59/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
/batch_normalization_33/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0k
&batch_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_33/batchnorm/addAddV27batch_normalization_33/batchnorm/ReadVariableOp:value:0/batch_normalization_33/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_33/batchnorm/RsqrtRsqrt(batch_normalization_33/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_33/batchnorm/mulMul*batch_normalization_33/batchnorm/Rsqrt:y:0;batch_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_33/batchnorm/mul_1Muldense_59/Relu:activations:0(batch_normalization_33/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
1batch_normalization_33/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_33_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_33/batchnorm/mul_2Mul9batch_normalization_33/batchnorm/ReadVariableOp_1:value:0(batch_normalization_33/batchnorm/mul:z:0*
T0*
_output_shapes
: �
1batch_normalization_33/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_33_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
$batch_normalization_33/batchnorm/subSub9batch_normalization_33/batchnorm/ReadVariableOp_2:value:0*batch_normalization_33/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_33/batchnorm/add_1AddV2*batch_normalization_33/batchnorm/mul_1:z:0(batch_normalization_33/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� }
dropout_51/IdentityIdentity*batch_normalization_33/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� �
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_60/MatMulMatMuldropout_51/Identity:output:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:��������� \
$token_and_position_embedding_8/ShapeShapeinputs_0*
T0*
_output_shapes
:�
2token_and_position_embedding_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������~
4token_and_position_embedding_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4token_and_position_embedding_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,token_and_position_embedding_8/strided_sliceStridedSlice-token_and_position_embedding_8/Shape:output:0;token_and_position_embedding_8/strided_slice/stack:output:0=token_and_position_embedding_8/strided_slice/stack_1:output:0=token_and_position_embedding_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*token_and_position_embedding_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : l
*token_and_position_embedding_8/range/limitConst*
_output_shapes
: *
dtype0*
value	B :	l
*token_and_position_embedding_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
$token_and_position_embedding_8/rangeRange3token_and_position_embedding_8/range/start:output:03token_and_position_embedding_8/range/limit:output:03token_and_position_embedding_8/range/delta:output:0*
_output_shapes
:	�
<token_and_position_embedding_8/embedding_17/embedding_lookupResourceGatherDtoken_and_position_embedding_8_embedding_17_embedding_lookup_9509945-token_and_position_embedding_8/range:output:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_17/embedding_lookup/9509945*
_output_shapes

:	*
dtype0�
Etoken_and_position_embedding_8/embedding_17/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_8/embedding_17/embedding_lookup:output:0*
T0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_17/embedding_lookup/9509945*
_output_shapes

:	�
Gtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:	�
0token_and_position_embedding_8/embedding_16/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������	�
<token_and_position_embedding_8/embedding_16/embedding_lookupResourceGatherDtoken_and_position_embedding_8_embedding_16_embedding_lookup_95099514token_and_position_embedding_8/embedding_16/Cast:y:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_16/embedding_lookup/9509951*+
_output_shapes
:���������	*
dtype0�
Etoken_and_position_embedding_8/embedding_16/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_8/embedding_16/embedding_lookup:output:0*
T0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_16/embedding_lookup/9509951*+
_output_shapes
:���������	�
Gtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
"token_and_position_embedding_8/addAddV2Ptoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1:output:0Ptoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������	�
/batch_normalization_34/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_34_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0k
&batch_normalization_34/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_34/batchnorm/addAddV27batch_normalization_34/batchnorm/ReadVariableOp:value:0/batch_normalization_34/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_34/batchnorm/RsqrtRsqrt(batch_normalization_34/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_34/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_34_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_34/batchnorm/mulMul*batch_normalization_34/batchnorm/Rsqrt:y:0;batch_normalization_34/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_34/batchnorm/mul_1Muldense_60/Relu:activations:0(batch_normalization_34/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
1batch_normalization_34/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_34_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_34/batchnorm/mul_2Mul9batch_normalization_34/batchnorm/ReadVariableOp_1:value:0(batch_normalization_34/batchnorm/mul:z:0*
T0*
_output_shapes
: �
1batch_normalization_34/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_34_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
$batch_normalization_34/batchnorm/subSub9batch_normalization_34/batchnorm/ReadVariableOp_2:value:0*batch_normalization_34/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_34/batchnorm/add_1AddV2*batch_normalization_34/batchnorm/mul_1:z:0(batch_normalization_34/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
@transformer_block_8/attention/query/einsum/Einsum/ReadVariableOpReadVariableOpItransformer_block_8_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
1transformer_block_8/attention/query/einsum/EinsumEinsum&token_and_position_embedding_8/add:z:0Htransformer_block_8/attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
6transformer_block_8/attention/query/add/ReadVariableOpReadVariableOp?transformer_block_8_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
'transformer_block_8/attention/query/addAddV2:transformer_block_8/attention/query/einsum/Einsum:output:0>transformer_block_8/attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
>transformer_block_8/attention/key/einsum/Einsum/ReadVariableOpReadVariableOpGtransformer_block_8_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
/transformer_block_8/attention/key/einsum/EinsumEinsum&token_and_position_embedding_8/add:z:0Ftransformer_block_8/attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
4transformer_block_8/attention/key/add/ReadVariableOpReadVariableOp=transformer_block_8_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
%transformer_block_8/attention/key/addAddV28transformer_block_8/attention/key/einsum/Einsum:output:0<transformer_block_8/attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
@transformer_block_8/attention/value/einsum/Einsum/ReadVariableOpReadVariableOpItransformer_block_8_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
1transformer_block_8/attention/value/einsum/EinsumEinsum&token_and_position_embedding_8/add:z:0Htransformer_block_8/attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������	*
equationabc,cde->abde�
6transformer_block_8/attention/value/add/ReadVariableOpReadVariableOp?transformer_block_8_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
'transformer_block_8/attention/value/addAddV2:transformer_block_8/attention/value/einsum/Einsum:output:0>transformer_block_8/attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	h
#transformer_block_8/attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
!transformer_block_8/attention/MulMul+transformer_block_8/attention/query/add:z:0,transformer_block_8/attention/Mul/y:output:0*
T0*/
_output_shapes
:���������	�
+transformer_block_8/attention/einsum/EinsumEinsum)transformer_block_8/attention/key/add:z:0%transformer_block_8/attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������		*
equationaecd,abcd->acbe�
-transformer_block_8/attention/softmax/SoftmaxSoftmax4transformer_block_8/attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������		�
.transformer_block_8/attention/dropout/IdentityIdentity7transformer_block_8/attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������		�
-transformer_block_8/attention/einsum_1/EinsumEinsum7transformer_block_8/attention/dropout/Identity:output:0+transformer_block_8/attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������	*
equationacbe,aecd->abcd�
Ktransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_8_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
<transformer_block_8/attention/attention_output/einsum/EinsumEinsum6transformer_block_8/attention/einsum_1/Einsum:output:0Stransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������	*
equationabcd,cde->abe�
Atransformer_block_8/attention/attention_output/add/ReadVariableOpReadVariableOpJtransformer_block_8_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
2transformer_block_8/attention/attention_output/addAddV2Etransformer_block_8/attention/attention_output/einsum/Einsum:output:0Itransformer_block_8/attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
'transformer_block_8/dropout_48/IdentityIdentity6transformer_block_8/attention/attention_output/add:z:0*
T0*+
_output_shapes
:���������	�
transformer_block_8/addAddV2&token_and_position_embedding_8/add:z:00transformer_block_8/dropout_48/Identity:output:0*
T0*+
_output_shapes
:���������	�
Itransformer_block_8/layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_block_8/layer_normalization_16/moments/meanMeantransformer_block_8/add:z:0Rtransformer_block_8/layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
?transformer_block_8/layer_normalization_16/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Dtransformer_block_8/layer_normalization_16/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add:z:0Htransformer_block_8/layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Mtransformer_block_8/layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_block_8/layer_normalization_16/moments/varianceMeanHtransformer_block_8/layer_normalization_16/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(
:transformer_block_8/layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
8transformer_block_8/layer_normalization_16/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_16/moments/variance:output:0Ctransformer_block_8/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_16/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
8transformer_block_8/layer_normalization_16/batchnorm/mulMul>transformer_block_8/layer_normalization_16/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_16/batchnorm/mul_1Multransformer_block_8/add:z:0<transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_16/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_16/moments/mean:output:0<transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
8transformer_block_8/layer_normalization_16/batchnorm/subSubKtransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_16/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_16/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
Btransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_8_sequential_8_dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
8transformer_block_8/sequential_8/dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block_8/sequential_8/dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
9transformer_block_8/sequential_8/dense_56/Tensordot/ShapeShape>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Atransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2GatherV2Btransformer_block_8/sequential_8/dense_56/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_56/Tensordot/free:output:0Jtransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ctransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>transformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1GatherV2Btransformer_block_8/sequential_8/dense_56/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_56/Tensordot/axes:output:0Ltransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
9transformer_block_8/sequential_8/dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
8transformer_block_8/sequential_8/dense_56/Tensordot/ProdProdEtransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2:output:0Btransformer_block_8/sequential_8/dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: �
;transformer_block_8/sequential_8/dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
:transformer_block_8/sequential_8/dense_56/Tensordot/Prod_1ProdGtransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2_1:output:0Dtransformer_block_8/sequential_8/dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
?transformer_block_8/sequential_8/dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:transformer_block_8/sequential_8/dense_56/Tensordot/concatConcatV2Atransformer_block_8/sequential_8/dense_56/Tensordot/free:output:0Atransformer_block_8/sequential_8/dense_56/Tensordot/axes:output:0Htransformer_block_8/sequential_8/dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
9transformer_block_8/sequential_8/dense_56/Tensordot/stackPackAtransformer_block_8/sequential_8/dense_56/Tensordot/Prod:output:0Ctransformer_block_8/sequential_8/dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
=transformer_block_8/sequential_8/dense_56/Tensordot/transpose	Transpose>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0Ctransformer_block_8/sequential_8/dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
;transformer_block_8/sequential_8/dense_56/Tensordot/ReshapeReshapeAtransformer_block_8/sequential_8/dense_56/Tensordot/transpose:y:0Btransformer_block_8/sequential_8/dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
:transformer_block_8/sequential_8/dense_56/Tensordot/MatMulMatMulDtransformer_block_8/sequential_8/dense_56/Tensordot/Reshape:output:0Jtransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;transformer_block_8/sequential_8/dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_block_8/sequential_8/dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<transformer_block_8/sequential_8/dense_56/Tensordot/concat_1ConcatV2Etransformer_block_8/sequential_8/dense_56/Tensordot/GatherV2:output:0Dtransformer_block_8/sequential_8/dense_56/Tensordot/Const_2:output:0Jtransformer_block_8/sequential_8/dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
3transformer_block_8/sequential_8/dense_56/TensordotReshapeDtransformer_block_8/sequential_8/dense_56/Tensordot/MatMul:product:0Etransformer_block_8/sequential_8/dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
@transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_8_sequential_8_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
1transformer_block_8/sequential_8/dense_56/BiasAddBiasAdd<transformer_block_8/sequential_8/dense_56/Tensordot:output:0Htransformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
.transformer_block_8/sequential_8/dense_56/ReluRelu:transformer_block_8/sequential_8/dense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
Btransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_8_sequential_8_dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
8transformer_block_8/sequential_8/dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block_8/sequential_8/dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
9transformer_block_8/sequential_8/dense_57/Tensordot/ShapeShape<transformer_block_8/sequential_8/dense_56/Relu:activations:0*
T0*
_output_shapes
:�
Atransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2GatherV2Btransformer_block_8/sequential_8/dense_57/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_57/Tensordot/free:output:0Jtransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ctransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>transformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1GatherV2Btransformer_block_8/sequential_8/dense_57/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_57/Tensordot/axes:output:0Ltransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
9transformer_block_8/sequential_8/dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
8transformer_block_8/sequential_8/dense_57/Tensordot/ProdProdEtransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2:output:0Btransformer_block_8/sequential_8/dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: �
;transformer_block_8/sequential_8/dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
:transformer_block_8/sequential_8/dense_57/Tensordot/Prod_1ProdGtransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2_1:output:0Dtransformer_block_8/sequential_8/dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
?transformer_block_8/sequential_8/dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:transformer_block_8/sequential_8/dense_57/Tensordot/concatConcatV2Atransformer_block_8/sequential_8/dense_57/Tensordot/free:output:0Atransformer_block_8/sequential_8/dense_57/Tensordot/axes:output:0Htransformer_block_8/sequential_8/dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
9transformer_block_8/sequential_8/dense_57/Tensordot/stackPackAtransformer_block_8/sequential_8/dense_57/Tensordot/Prod:output:0Ctransformer_block_8/sequential_8/dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
=transformer_block_8/sequential_8/dense_57/Tensordot/transpose	Transpose<transformer_block_8/sequential_8/dense_56/Relu:activations:0Ctransformer_block_8/sequential_8/dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
;transformer_block_8/sequential_8/dense_57/Tensordot/ReshapeReshapeAtransformer_block_8/sequential_8/dense_57/Tensordot/transpose:y:0Btransformer_block_8/sequential_8/dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
:transformer_block_8/sequential_8/dense_57/Tensordot/MatMulMatMulDtransformer_block_8/sequential_8/dense_57/Tensordot/Reshape:output:0Jtransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;transformer_block_8/sequential_8/dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_block_8/sequential_8/dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<transformer_block_8/sequential_8/dense_57/Tensordot/concat_1ConcatV2Etransformer_block_8/sequential_8/dense_57/Tensordot/GatherV2:output:0Dtransformer_block_8/sequential_8/dense_57/Tensordot/Const_2:output:0Jtransformer_block_8/sequential_8/dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
3transformer_block_8/sequential_8/dense_57/TensordotReshapeDtransformer_block_8/sequential_8/dense_57/Tensordot/MatMul:product:0Etransformer_block_8/sequential_8/dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
@transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_8_sequential_8_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
1transformer_block_8/sequential_8/dense_57/BiasAddBiasAdd<transformer_block_8/sequential_8/dense_57/Tensordot:output:0Htransformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
'transformer_block_8/dropout_49/IdentityIdentity:transformer_block_8/sequential_8/dense_57/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
transformer_block_8/add_1AddV2>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:00transformer_block_8/dropout_49/Identity:output:0*
T0*+
_output_shapes
:���������	�
Itransformer_block_8/layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_block_8/layer_normalization_17/moments/meanMeantransformer_block_8/add_1:z:0Rtransformer_block_8/layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
?transformer_block_8/layer_normalization_17/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
Dtransformer_block_8/layer_normalization_17/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add_1:z:0Htransformer_block_8/layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
Mtransformer_block_8/layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_block_8/layer_normalization_17/moments/varianceMeanHtransformer_block_8/layer_normalization_17/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(
:transformer_block_8/layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
8transformer_block_8/layer_normalization_17/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_17/moments/variance:output:0Ctransformer_block_8/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_17/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
8transformer_block_8/layer_normalization_17/batchnorm/mulMul>transformer_block_8/layer_normalization_17/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_17/batchnorm/mul_1Multransformer_block_8/add_1:z:0<transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_17/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_17/moments/mean:output:0<transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
8transformer_block_8/layer_normalization_17/batchnorm/subSubKtransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
:transformer_block_8/layer_normalization_17/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_17/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	}
dropout_52/IdentityIdentity*batch_normalization_34/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� s
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d_8/MeanMean>transformer_block_8/layer_normalization_17/batchnorm/add_1:z:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_61/MatMulMatMuldropout_52/Identity:output:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_58/MatMulMatMul(global_average_pooling1d_8/Mean:output:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
/batch_normalization_35/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_35_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0k
&batch_normalization_35/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_35/batchnorm/addAddV27batch_normalization_35/batchnorm/ReadVariableOp:value:0/batch_normalization_35/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_35/batchnorm/RsqrtRsqrt(batch_normalization_35/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_35/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_35_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_35/batchnorm/mulMul*batch_normalization_35/batchnorm/Rsqrt:y:0;batch_normalization_35/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_35/batchnorm/mul_1Muldense_61/Relu:activations:0(batch_normalization_35/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
1batch_normalization_35/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_35_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_35/batchnorm/mul_2Mul9batch_normalization_35/batchnorm/ReadVariableOp_1:value:0(batch_normalization_35/batchnorm/mul:z:0*
T0*
_output_shapes
: �
1batch_normalization_35/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_35_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
$batch_normalization_35/batchnorm/subSub9batch_normalization_35/batchnorm/ReadVariableOp_2:value:0*batch_normalization_35/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_35/batchnorm/add_1AddV2*batch_normalization_35/batchnorm/mul_1:z:0(batch_normalization_35/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
/batch_normalization_32/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0k
&batch_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_32/batchnorm/addAddV27batch_normalization_32/batchnorm/ReadVariableOp:value:0/batch_normalization_32/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_32/batchnorm/RsqrtRsqrt(batch_normalization_32/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_32/batchnorm/mulMul*batch_normalization_32/batchnorm/Rsqrt:y:0;batch_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_32/batchnorm/mul_1Muldense_58/Relu:activations:0(batch_normalization_32/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
1batch_normalization_32/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_32_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_32/batchnorm/mul_2Mul9batch_normalization_32/batchnorm/ReadVariableOp_1:value:0(batch_normalization_32/batchnorm/mul:z:0*
T0*
_output_shapes
: �
1batch_normalization_32/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_32_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
$batch_normalization_32/batchnorm/subSub9batch_normalization_32/batchnorm/ReadVariableOp_2:value:0*batch_normalization_32/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_32/batchnorm/add_1AddV2*batch_normalization_32/batchnorm/mul_1:z:0(batch_normalization_32/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� }
dropout_50/IdentityIdentity*batch_normalization_32/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� }
dropout_53/IdentityIdentity*batch_normalization_35/batchnorm/add_1:z:0*
T0*'
_output_shapes
:��������� [
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_8/concatConcatV2dropout_50/Identity:output:0dropout_53/Identity:output:0"concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_62/MatMulMatMulconcatenate_8/concat:output:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_62/SoftmaxSoftmaxdense_62/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_62/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_32/batchnorm/ReadVariableOp2^batch_normalization_32/batchnorm/ReadVariableOp_12^batch_normalization_32/batchnorm/ReadVariableOp_24^batch_normalization_32/batchnorm/mul/ReadVariableOp0^batch_normalization_33/batchnorm/ReadVariableOp2^batch_normalization_33/batchnorm/ReadVariableOp_12^batch_normalization_33/batchnorm/ReadVariableOp_24^batch_normalization_33/batchnorm/mul/ReadVariableOp0^batch_normalization_34/batchnorm/ReadVariableOp2^batch_normalization_34/batchnorm/ReadVariableOp_12^batch_normalization_34/batchnorm/ReadVariableOp_24^batch_normalization_34/batchnorm/mul/ReadVariableOp0^batch_normalization_35/batchnorm/ReadVariableOp2^batch_normalization_35/batchnorm/ReadVariableOp_12^batch_normalization_35/batchnorm/ReadVariableOp_24^batch_normalization_35/batchnorm/mul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp=^token_and_position_embedding_8/embedding_16/embedding_lookup=^token_and_position_embedding_8/embedding_17/embedding_lookupB^transformer_block_8/attention/attention_output/add/ReadVariableOpL^transformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp5^transformer_block_8/attention/key/add/ReadVariableOp?^transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp7^transformer_block_8/attention/query/add/ReadVariableOpA^transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp7^transformer_block_8/attention/value/add/ReadVariableOpA^transformer_block_8/attention/value/einsum/Einsum/ReadVariableOpD^transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpD^transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpA^transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOpC^transformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpA^transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOpC^transformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_32/batchnorm/ReadVariableOp/batch_normalization_32/batchnorm/ReadVariableOp2f
1batch_normalization_32/batchnorm/ReadVariableOp_11batch_normalization_32/batchnorm/ReadVariableOp_12f
1batch_normalization_32/batchnorm/ReadVariableOp_21batch_normalization_32/batchnorm/ReadVariableOp_22j
3batch_normalization_32/batchnorm/mul/ReadVariableOp3batch_normalization_32/batchnorm/mul/ReadVariableOp2b
/batch_normalization_33/batchnorm/ReadVariableOp/batch_normalization_33/batchnorm/ReadVariableOp2f
1batch_normalization_33/batchnorm/ReadVariableOp_11batch_normalization_33/batchnorm/ReadVariableOp_12f
1batch_normalization_33/batchnorm/ReadVariableOp_21batch_normalization_33/batchnorm/ReadVariableOp_22j
3batch_normalization_33/batchnorm/mul/ReadVariableOp3batch_normalization_33/batchnorm/mul/ReadVariableOp2b
/batch_normalization_34/batchnorm/ReadVariableOp/batch_normalization_34/batchnorm/ReadVariableOp2f
1batch_normalization_34/batchnorm/ReadVariableOp_11batch_normalization_34/batchnorm/ReadVariableOp_12f
1batch_normalization_34/batchnorm/ReadVariableOp_21batch_normalization_34/batchnorm/ReadVariableOp_22j
3batch_normalization_34/batchnorm/mul/ReadVariableOp3batch_normalization_34/batchnorm/mul/ReadVariableOp2b
/batch_normalization_35/batchnorm/ReadVariableOp/batch_normalization_35/batchnorm/ReadVariableOp2f
1batch_normalization_35/batchnorm/ReadVariableOp_11batch_normalization_35/batchnorm/ReadVariableOp_12f
1batch_normalization_35/batchnorm/ReadVariableOp_21batch_normalization_35/batchnorm/ReadVariableOp_22j
3batch_normalization_35/batchnorm/mul/ReadVariableOp3batch_normalization_35/batchnorm/mul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2|
<token_and_position_embedding_8/embedding_16/embedding_lookup<token_and_position_embedding_8/embedding_16/embedding_lookup2|
<token_and_position_embedding_8/embedding_17/embedding_lookup<token_and_position_embedding_8/embedding_17/embedding_lookup2�
Atransformer_block_8/attention/attention_output/add/ReadVariableOpAtransformer_block_8/attention/attention_output/add/ReadVariableOp2�
Ktransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOpKtransformer_block_8/attention/attention_output/einsum/Einsum/ReadVariableOp2l
4transformer_block_8/attention/key/add/ReadVariableOp4transformer_block_8/attention/key/add/ReadVariableOp2�
>transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp>transformer_block_8/attention/key/einsum/Einsum/ReadVariableOp2p
6transformer_block_8/attention/query/add/ReadVariableOp6transformer_block_8/attention/query/add/ReadVariableOp2�
@transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp@transformer_block_8/attention/query/einsum/Einsum/ReadVariableOp2p
6transformer_block_8/attention/value/add/ReadVariableOp6transformer_block_8/attention/value/add/ReadVariableOp2�
@transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp@transformer_block_8/attention/value/einsum/Einsum/ReadVariableOp2�
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp2�
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp2�
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp2�
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp2�
@transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp@transformer_block_8/sequential_8/dense_56/BiasAdd/ReadVariableOp2�
Btransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOpBtransformer_block_8/sequential_8/dense_56/Tensordot/ReadVariableOp2�
@transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp@transformer_block_8/sequential_8/dense_57/BiasAdd/ReadVariableOp2�
Btransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOpBtransformer_block_8/sequential_8/dense_57/Tensordot/ReadVariableOp:Q M
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
�?
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_9511616

inputs<
*dense_56_tensordot_readvariableop_resource:6
(dense_56_biasadd_readvariableop_resource:<
*dense_57_tensordot_readvariableop_resource:6
(dense_57_biasadd_readvariableop_resource:
identity��dense_56/BiasAdd/ReadVariableOp�!dense_56/Tensordot/ReadVariableOp�dense_57/BiasAdd/ReadVariableOp�!dense_57/Tensordot/ReadVariableOp�
!dense_56/Tensordot/ReadVariableOpReadVariableOp*dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_56/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_56/Tensordot/GatherV2GatherV2!dense_56/Tensordot/Shape:output:0 dense_56/Tensordot/free:output:0)dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_56/Tensordot/GatherV2_1GatherV2!dense_56/Tensordot/Shape:output:0 dense_56/Tensordot/axes:output:0+dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_56/Tensordot/ProdProd$dense_56/Tensordot/GatherV2:output:0!dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_56/Tensordot/Prod_1Prod&dense_56/Tensordot/GatherV2_1:output:0#dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_56/Tensordot/concatConcatV2 dense_56/Tensordot/free:output:0 dense_56/Tensordot/axes:output:0'dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_56/Tensordot/stackPack dense_56/Tensordot/Prod:output:0"dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_56/Tensordot/transpose	Transposeinputs"dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
dense_56/Tensordot/ReshapeReshape dense_56/Tensordot/transpose:y:0!dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_56/Tensordot/MatMulMatMul#dense_56/Tensordot/Reshape:output:0)dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_56/Tensordot/concat_1ConcatV2$dense_56/Tensordot/GatherV2:output:0#dense_56/Tensordot/Const_2:output:0)dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_56/TensordotReshape#dense_56/Tensordot/MatMul:product:0$dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_56/BiasAddBiasAdddense_56/Tensordot:output:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	f
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
!dense_57/Tensordot/ReadVariableOpReadVariableOp*dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_57/Tensordot/ShapeShapedense_56/Relu:activations:0*
T0*
_output_shapes
:b
 dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_57/Tensordot/GatherV2GatherV2!dense_57/Tensordot/Shape:output:0 dense_57/Tensordot/free:output:0)dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_57/Tensordot/GatherV2_1GatherV2!dense_57/Tensordot/Shape:output:0 dense_57/Tensordot/axes:output:0+dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_57/Tensordot/ProdProd$dense_57/Tensordot/GatherV2:output:0!dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_57/Tensordot/Prod_1Prod&dense_57/Tensordot/GatherV2_1:output:0#dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_57/Tensordot/concatConcatV2 dense_57/Tensordot/free:output:0 dense_57/Tensordot/axes:output:0'dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_57/Tensordot/stackPack dense_57/Tensordot/Prod:output:0"dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_57/Tensordot/transpose	Transposedense_56/Relu:activations:0"dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
dense_57/Tensordot/ReshapeReshape dense_57/Tensordot/transpose:y:0!dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_57/Tensordot/MatMulMatMul#dense_57/Tensordot/Reshape:output:0)dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_57/Tensordot/concat_1ConcatV2$dense_57/Tensordot/GatherV2:output:0#dense_57/Tensordot/Const_2:output:0)dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_57/TensordotReshape#dense_57/Tensordot/MatMul:product:0$dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_57/BiasAddBiasAdddense_57/Tensordot:output:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	l
IdentityIdentitydense_57/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp ^dense_56/BiasAdd/ReadVariableOp"^dense_56/Tensordot/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp"^dense_57/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2F
!dense_56/Tensordot/ReadVariableOp!dense_56/Tensordot/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2F
!dense_57/Tensordot/ReadVariableOp!dense_57/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�f
�
D__inference_model_8_layer_call_and_return_conditional_losses_9509707
input_17
input_18"
dense_59_9509601:
 
dense_59_9509603: ,
batch_normalization_33_9509606: ,
batch_normalization_33_9509608: ,
batch_normalization_33_9509610: ,
batch_normalization_33_9509612: "
dense_60_9509616:  
dense_60_9509618: 8
&token_and_position_embedding_8_9509621:	8
&token_and_position_embedding_8_9509623:,
batch_normalization_34_9509626: ,
batch_normalization_34_9509628: ,
batch_normalization_34_9509630: ,
batch_normalization_34_9509632: 1
transformer_block_8_9509635:-
transformer_block_8_9509637:1
transformer_block_8_9509639:-
transformer_block_8_9509641:1
transformer_block_8_9509643:-
transformer_block_8_9509645:1
transformer_block_8_9509647:)
transformer_block_8_9509649:)
transformer_block_8_9509651:)
transformer_block_8_9509653:-
transformer_block_8_9509655:)
transformer_block_8_9509657:-
transformer_block_8_9509659:)
transformer_block_8_9509661:)
transformer_block_8_9509663:)
transformer_block_8_9509665:"
dense_61_9509670:  
dense_61_9509672: "
dense_58_9509675: 
dense_58_9509677: ,
batch_normalization_35_9509680: ,
batch_normalization_35_9509682: ,
batch_normalization_35_9509684: ,
batch_normalization_35_9509686: ,
batch_normalization_32_9509689: ,
batch_normalization_32_9509691: ,
batch_normalization_32_9509693: ,
batch_normalization_32_9509695: "
dense_62_9509701:@
dense_62_9509703:
identity��.batch_normalization_32/StatefulPartitionedCall�.batch_normalization_33/StatefulPartitionedCall�.batch_normalization_34/StatefulPartitionedCall�.batch_normalization_35/StatefulPartitionedCall� dense_58/StatefulPartitionedCall� dense_59/StatefulPartitionedCall� dense_60/StatefulPartitionedCall� dense_61/StatefulPartitionedCall� dense_62/StatefulPartitionedCall�"dropout_50/StatefulPartitionedCall�"dropout_51/StatefulPartitionedCall�"dropout_52/StatefulPartitionedCall�"dropout_53/StatefulPartitionedCall�6token_and_position_embedding_8/StatefulPartitionedCall�+transformer_block_8/StatefulPartitionedCall�
 dense_59/StatefulPartitionedCallStatefulPartitionedCallinput_18dense_59_9509601dense_59_9509603*
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
E__inference_dense_59_layer_call_and_return_conditional_losses_9508289�
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0batch_normalization_33_9509606batch_normalization_33_9509608batch_normalization_33_9509610batch_normalization_33_9509612*
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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9507806�
"dropout_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_9509082�
 dense_60/StatefulPartitionedCallStatefulPartitionedCall+dropout_51/StatefulPartitionedCall:output:0dense_60_9509616dense_60_9509618*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_9508322�
6token_and_position_embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_17&token_and_position_embedding_8_9509621&token_and_position_embedding_8_9509623*
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
GPU 2J 8� *d
f_R]
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9508353�
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_34_9509626batch_normalization_34_9509628batch_normalization_34_9509630batch_normalization_34_9509632*
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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9508081�
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_8/StatefulPartitionedCall:output:0transformer_block_8_9509635transformer_block_8_9509637transformer_block_8_9509639transformer_block_8_9509641transformer_block_8_9509643transformer_block_8_9509645transformer_block_8_9509647transformer_block_8_9509649transformer_block_8_9509651transformer_block_8_9509653transformer_block_8_9509655transformer_block_8_9509657transformer_block_8_9509659transformer_block_8_9509661transformer_block_8_9509663transformer_block_8_9509665*
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
GPU 2J 8� *Y
fTRR
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9509007�
"dropout_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0#^dropout_51/StatefulPartitionedCall*
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
G__inference_dropout_52_layer_call_and_return_conditional_losses_9508824�
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *`
f[RY
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9508102�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall+dropout_52/StatefulPartitionedCall:output:0dense_61_9509670dense_61_9509672*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_9508548�
 dense_58/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_58_9509675dense_58_9509677*
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
E__inference_dense_58_layer_call_and_return_conditional_losses_9508565�
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_35_9509680batch_normalization_35_9509682batch_normalization_35_9509684batch_normalization_35_9509686*
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9508258�
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_32_9509689batch_normalization_32_9509691batch_normalization_32_9509693batch_normalization_32_9509695*
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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9508176�
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0#^dropout_52/StatefulPartitionedCall*
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_9508781�
"dropout_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0#^dropout_50/StatefulPartitionedCall*
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
G__inference_dropout_53_layer_call_and_return_conditional_losses_9508758�
concatenate_8/PartitionedCallPartitionedCall+dropout_50/StatefulPartitionedCall:output:0+dropout_53/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9508610�
 dense_62/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_62_9509701dense_62_9509703*
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
GPU 2J 8� *N
fIRG
E__inference_dense_62_layer_call_and_return_conditional_losses_9508623x
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall#^dropout_51/StatefulPartitionedCall#^dropout_52/StatefulPartitionedCall#^dropout_53/StatefulPartitionedCall7^token_and_position_embedding_8/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2H
"dropout_51/StatefulPartitionedCall"dropout_51/StatefulPartitionedCall2H
"dropout_52/StatefulPartitionedCall"dropout_52/StatefulPartitionedCall2H
"dropout_53/StatefulPartitionedCall"dropout_53/StatefulPartitionedCall2p
6token_and_position_embedding_8/StatefulPartitionedCall6token_and_position_embedding_8/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_17:QM
'
_output_shapes
:���������

"
_user_specified_name
input_18
�
�
*__inference_dense_61_layer_call_fn_9511275

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
GPU 2J 8� *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_9508548o
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
�

�
E__inference_dense_61_layer_call_and_return_conditional_losses_9511286

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
E__inference_dense_58_layer_call_and_return_conditional_losses_9511266

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
�	
f
G__inference_dropout_51_layer_call_and_return_conditional_losses_9509082

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
�f
�
D__inference_model_8_layer_call_and_return_conditional_losses_9509302

inputs
inputs_1"
dense_59_9509196:
 
dense_59_9509198: ,
batch_normalization_33_9509201: ,
batch_normalization_33_9509203: ,
batch_normalization_33_9509205: ,
batch_normalization_33_9509207: "
dense_60_9509211:  
dense_60_9509213: 8
&token_and_position_embedding_8_9509216:	8
&token_and_position_embedding_8_9509218:,
batch_normalization_34_9509221: ,
batch_normalization_34_9509223: ,
batch_normalization_34_9509225: ,
batch_normalization_34_9509227: 1
transformer_block_8_9509230:-
transformer_block_8_9509232:1
transformer_block_8_9509234:-
transformer_block_8_9509236:1
transformer_block_8_9509238:-
transformer_block_8_9509240:1
transformer_block_8_9509242:)
transformer_block_8_9509244:)
transformer_block_8_9509246:)
transformer_block_8_9509248:-
transformer_block_8_9509250:)
transformer_block_8_9509252:-
transformer_block_8_9509254:)
transformer_block_8_9509256:)
transformer_block_8_9509258:)
transformer_block_8_9509260:"
dense_61_9509265:  
dense_61_9509267: "
dense_58_9509270: 
dense_58_9509272: ,
batch_normalization_35_9509275: ,
batch_normalization_35_9509277: ,
batch_normalization_35_9509279: ,
batch_normalization_35_9509281: ,
batch_normalization_32_9509284: ,
batch_normalization_32_9509286: ,
batch_normalization_32_9509288: ,
batch_normalization_32_9509290: "
dense_62_9509296:@
dense_62_9509298:
identity��.batch_normalization_32/StatefulPartitionedCall�.batch_normalization_33/StatefulPartitionedCall�.batch_normalization_34/StatefulPartitionedCall�.batch_normalization_35/StatefulPartitionedCall� dense_58/StatefulPartitionedCall� dense_59/StatefulPartitionedCall� dense_60/StatefulPartitionedCall� dense_61/StatefulPartitionedCall� dense_62/StatefulPartitionedCall�"dropout_50/StatefulPartitionedCall�"dropout_51/StatefulPartitionedCall�"dropout_52/StatefulPartitionedCall�"dropout_53/StatefulPartitionedCall�6token_and_position_embedding_8/StatefulPartitionedCall�+transformer_block_8/StatefulPartitionedCall�
 dense_59/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_59_9509196dense_59_9509198*
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
E__inference_dense_59_layer_call_and_return_conditional_losses_9508289�
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0batch_normalization_33_9509201batch_normalization_33_9509203batch_normalization_33_9509205batch_normalization_33_9509207*
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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9507806�
"dropout_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_9509082�
 dense_60/StatefulPartitionedCallStatefulPartitionedCall+dropout_51/StatefulPartitionedCall:output:0dense_60_9509211dense_60_9509213*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_9508322�
6token_and_position_embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputs&token_and_position_embedding_8_9509216&token_and_position_embedding_8_9509218*
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
GPU 2J 8� *d
f_R]
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9508353�
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_34_9509221batch_normalization_34_9509223batch_normalization_34_9509225batch_normalization_34_9509227*
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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9508081�
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_8/StatefulPartitionedCall:output:0transformer_block_8_9509230transformer_block_8_9509232transformer_block_8_9509234transformer_block_8_9509236transformer_block_8_9509238transformer_block_8_9509240transformer_block_8_9509242transformer_block_8_9509244transformer_block_8_9509246transformer_block_8_9509248transformer_block_8_9509250transformer_block_8_9509252transformer_block_8_9509254transformer_block_8_9509256transformer_block_8_9509258transformer_block_8_9509260*
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
GPU 2J 8� *Y
fTRR
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9509007�
"dropout_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0#^dropout_51/StatefulPartitionedCall*
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
G__inference_dropout_52_layer_call_and_return_conditional_losses_9508824�
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *`
f[RY
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9508102�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall+dropout_52/StatefulPartitionedCall:output:0dense_61_9509265dense_61_9509267*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_9508548�
 dense_58/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_58_9509270dense_58_9509272*
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
E__inference_dense_58_layer_call_and_return_conditional_losses_9508565�
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_35_9509275batch_normalization_35_9509277batch_normalization_35_9509279batch_normalization_35_9509281*
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9508258�
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_32_9509284batch_normalization_32_9509286batch_normalization_32_9509288batch_normalization_32_9509290*
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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9508176�
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0#^dropout_52/StatefulPartitionedCall*
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_9508781�
"dropout_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0#^dropout_50/StatefulPartitionedCall*
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
G__inference_dropout_53_layer_call_and_return_conditional_losses_9508758�
concatenate_8/PartitionedCallPartitionedCall+dropout_50/StatefulPartitionedCall:output:0+dropout_53/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9508610�
 dense_62/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_62_9509296dense_62_9509298*
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
GPU 2J 8� *N
fIRG
E__inference_dense_62_layer_call_and_return_conditional_losses_9508623x
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall#^dropout_51/StatefulPartitionedCall#^dropout_52/StatefulPartitionedCall#^dropout_53/StatefulPartitionedCall7^token_and_position_embedding_8/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
~:���������	:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2H
"dropout_51/StatefulPartitionedCall"dropout_51/StatefulPartitionedCall2H
"dropout_52/StatefulPartitionedCall"dropout_52/StatefulPartitionedCall2H
"dropout_53/StatefulPartitionedCall"dropout_53/StatefulPartitionedCall2p
6token_and_position_embedding_8/StatefulPartitionedCall6token_and_position_embedding_8/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������

 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9510706

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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9511208

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
8__inference_batch_normalization_34_layer_call_fn_9511141

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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9508034o
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
�
s
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9511219

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
�
e
G__inference_dropout_50_layer_call_and_return_conditional_losses_9508594

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
�
e
G__inference_dropout_52_layer_call_and_return_conditional_losses_9508534

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
�%
�
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9508258

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
@__inference_token_and_position_embedding_8_layer_call_fn_9510742
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
GPU 2J 8� *d
f_R]
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9508353s
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
��
�
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9509007

inputsK
5attention_query_einsum_einsum_readvariableop_resource:=
+attention_query_add_readvariableop_resource:I
3attention_key_einsum_einsum_readvariableop_resource:;
)attention_key_add_readvariableop_resource:K
5attention_value_einsum_einsum_readvariableop_resource:=
+attention_value_add_readvariableop_resource:V
@attention_attention_output_einsum_einsum_readvariableop_resource:D
6attention_attention_output_add_readvariableop_resource:J
<layer_normalization_16_batchnorm_mul_readvariableop_resource:F
8layer_normalization_16_batchnorm_readvariableop_resource:I
7sequential_8_dense_56_tensordot_readvariableop_resource:C
5sequential_8_dense_56_biasadd_readvariableop_resource:I
7sequential_8_dense_57_tensordot_readvariableop_resource:C
5sequential_8_dense_57_biasadd_readvariableop_resource:J
<layer_normalization_17_batchnorm_mul_readvariableop_resource:F
8layer_normalization_17_batchnorm_readvariableop_resource:
identity��-attention/attention_output/add/ReadVariableOp�7attention/attention_output/einsum/Einsum/ReadVariableOp� attention/key/add/ReadVariableOp�*attention/key/einsum/Einsum/ReadVariableOp�"attention/query/add/ReadVariableOp�,attention/query/einsum/Einsum/ReadVariableOp�"attention/value/add/ReadVariableOp�,attention/value/einsum/Einsum/ReadVariableOp�/layer_normalization_16/batchnorm/ReadVariableOp�3layer_normalization_16/batchnorm/mul/ReadVariableOp�/layer_normalization_17/batchnorm/ReadVariableOp�3layer_normalization_17/batchnorm/mul/ReadVariableOp�,sequential_8/dense_56/BiasAdd/ReadVariableOp�.sequential_8/dense_56/Tensordot/ReadVariableOp�,sequential_8/dense_57/BiasAdd/ReadVariableOp�.sequential_8/dense_57/Tensordot/ReadVariableOp�
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
:���������	]
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_48/dropout/MulMul"attention/attention_output/add:z:0!dropout_48/dropout/Const:output:0*
T0*+
_output_shapes
:���������	j
dropout_48/dropout/ShapeShape"attention/attention_output/add:z:0*
T0*
_output_shapes
:�
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0f
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	h
addAddV2inputsdropout_48/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_16/moments/meanMeanadd:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(k
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/mul_1Muladd:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	�
.sequential_8/dense_56/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_56_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_8/dense_56/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_56/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_8/dense_56/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_8/dense_56/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_56/Tensordot/GatherV2GatherV2.sequential_8/dense_56/Tensordot/Shape:output:0-sequential_8/dense_56/Tensordot/free:output:06sequential_8/dense_56/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_56/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_8/dense_56/Tensordot/GatherV2_1GatherV2.sequential_8/dense_56/Tensordot/Shape:output:0-sequential_8/dense_56/Tensordot/axes:output:08sequential_8/dense_56/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_56/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_8/dense_56/Tensordot/ProdProd1sequential_8/dense_56/Tensordot/GatherV2:output:0.sequential_8/dense_56/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_56/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_8/dense_56/Tensordot/Prod_1Prod3sequential_8/dense_56/Tensordot/GatherV2_1:output:00sequential_8/dense_56/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_56/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_8/dense_56/Tensordot/concatConcatV2-sequential_8/dense_56/Tensordot/free:output:0-sequential_8/dense_56/Tensordot/axes:output:04sequential_8/dense_56/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_8/dense_56/Tensordot/stackPack-sequential_8/dense_56/Tensordot/Prod:output:0/sequential_8/dense_56/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_8/dense_56/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_8/dense_56/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
'sequential_8/dense_56/Tensordot/ReshapeReshape-sequential_8/dense_56/Tensordot/transpose:y:0.sequential_8/dense_56/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_8/dense_56/Tensordot/MatMulMatMul0sequential_8/dense_56/Tensordot/Reshape:output:06sequential_8/dense_56/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
'sequential_8/dense_56/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_8/dense_56/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_56/Tensordot/concat_1ConcatV21sequential_8/dense_56/Tensordot/GatherV2:output:00sequential_8/dense_56/Tensordot/Const_2:output:06sequential_8/dense_56/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_8/dense_56/TensordotReshape0sequential_8/dense_56/Tensordot/MatMul:product:01sequential_8/dense_56/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
,sequential_8/dense_56/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_8/dense_56/BiasAddBiasAdd(sequential_8/dense_56/Tensordot:output:04sequential_8/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
sequential_8/dense_56/ReluRelu&sequential_8/dense_56/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
.sequential_8/dense_57/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_57_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_8/dense_57/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_57/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_8/dense_57/Tensordot/ShapeShape(sequential_8/dense_56/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_8/dense_57/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_57/Tensordot/GatherV2GatherV2.sequential_8/dense_57/Tensordot/Shape:output:0-sequential_8/dense_57/Tensordot/free:output:06sequential_8/dense_57/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_57/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_8/dense_57/Tensordot/GatherV2_1GatherV2.sequential_8/dense_57/Tensordot/Shape:output:0-sequential_8/dense_57/Tensordot/axes:output:08sequential_8/dense_57/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_57/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_8/dense_57/Tensordot/ProdProd1sequential_8/dense_57/Tensordot/GatherV2:output:0.sequential_8/dense_57/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_57/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_8/dense_57/Tensordot/Prod_1Prod3sequential_8/dense_57/Tensordot/GatherV2_1:output:00sequential_8/dense_57/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_57/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_8/dense_57/Tensordot/concatConcatV2-sequential_8/dense_57/Tensordot/free:output:0-sequential_8/dense_57/Tensordot/axes:output:04sequential_8/dense_57/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_8/dense_57/Tensordot/stackPack-sequential_8/dense_57/Tensordot/Prod:output:0/sequential_8/dense_57/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_8/dense_57/Tensordot/transpose	Transpose(sequential_8/dense_56/Relu:activations:0/sequential_8/dense_57/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
'sequential_8/dense_57/Tensordot/ReshapeReshape-sequential_8/dense_57/Tensordot/transpose:y:0.sequential_8/dense_57/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_8/dense_57/Tensordot/MatMulMatMul0sequential_8/dense_57/Tensordot/Reshape:output:06sequential_8/dense_57/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
'sequential_8/dense_57/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_8/dense_57/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_8/dense_57/Tensordot/concat_1ConcatV21sequential_8/dense_57/Tensordot/GatherV2:output:00sequential_8/dense_57/Tensordot/Const_2:output:06sequential_8/dense_57/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_8/dense_57/TensordotReshape0sequential_8/dense_57/Tensordot/MatMul:product:01sequential_8/dense_57/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
,sequential_8/dense_57/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_8/dense_57/BiasAddBiasAdd(sequential_8/dense_57/Tensordot:output:04sequential_8/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	]
dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_49/dropout/MulMul&sequential_8/dense_57/BiasAdd:output:0!dropout_49/dropout/Const:output:0*
T0*+
_output_shapes
:���������	n
dropout_49/dropout/ShapeShape&sequential_8/dense_57/BiasAdd:output:0*
T0*
_output_shapes
:�
/dropout_49/dropout/random_uniform/RandomUniformRandomUniform!dropout_49/dropout/Shape:output:0*
T0*+
_output_shapes
:���������	*
dtype0f
!dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_49/dropout/GreaterEqualGreaterEqual8dropout_49/dropout/random_uniform/RandomUniform:output:0*dropout_49/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������	�
dropout_49/dropout/CastCast#dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������	�
dropout_49/dropout/Mul_1Muldropout_49/dropout/Mul:z:0dropout_49/dropout/Cast:y:0*
T0*+
_output_shapes
:���������	�
add_1AddV2*layer_normalization_16/batchnorm/add_1:z:0dropout_49/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������	
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_17/moments/meanMean	add_1:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(�
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������	�
0layer_normalization_17/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������	�
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������	*
	keep_dims(k
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������	�
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������	�
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������	�
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������	}
IdentityIdentity*layer_normalization_17/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp-^sequential_8/dense_56/BiasAdd/ReadVariableOp/^sequential_8/dense_56/Tensordot/ReadVariableOp-^sequential_8/dense_57/BiasAdd/ReadVariableOp/^sequential_8/dense_57/Tensordot/ReadVariableOp*"
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
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2\
,sequential_8/dense_56/BiasAdd/ReadVariableOp,sequential_8/dense_56/BiasAdd/ReadVariableOp2`
.sequential_8/dense_56/Tensordot/ReadVariableOp.sequential_8/dense_56/Tensordot/ReadVariableOp2\
,sequential_8/dense_57/BiasAdd/ReadVariableOp,sequential_8/dense_57/BiasAdd/ReadVariableOp2`
.sequential_8/dense_57/Tensordot/ReadVariableOp.sequential_8/dense_57/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�

)__inference_model_8_layer_call_fn_9509807
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
GPU 2J 8� *M
fHRF
D__inference_model_8_layer_call_and_return_conditional_losses_9508630o
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
�
�
E__inference_dense_56_layer_call_and_return_conditional_losses_9507855

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
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507996
dense_56_input"
dense_56_9507985:
dense_56_9507987:"
dense_57_9507990:
dense_57_9507992:
identity�� dense_56/StatefulPartitionedCall� dense_57/StatefulPartitionedCall�
 dense_56/StatefulPartitionedCallStatefulPartitionedCalldense_56_inputdense_56_9507985dense_56_9507987*
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
E__inference_dense_56_layer_call_and_return_conditional_losses_9507855�
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_9507990dense_57_9507992*
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
E__inference_dense_57_layer_call_and_return_conditional_losses_9507891|
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:[ W
+
_output_shapes
:���������	
(
_user_specified_namedense_56_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_171
serving_default_input_17:0���������	
=
input_181
serving_default_input_18:0���������
<
dense_620
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
)__inference_model_8_layer_call_fn_9508721
)__inference_model_8_layer_call_fn_9509807
)__inference_model_8_layer_call_fn_9509901
)__inference_model_8_layer_call_fn_9509487�
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
D__inference_model_8_layer_call_and_return_conditional_losses_9510157
D__inference_model_8_layer_call_and_return_conditional_losses_9510510
D__inference_model_8_layer_call_and_return_conditional_losses_9509597
D__inference_model_8_layer_call_and_return_conditional_losses_9509707�
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
"__inference__wrapped_model_9507735input_17input_18"�
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
 2dense_59/kernel
: 2dense_59/bias
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
*__inference_dense_59_layer_call_fn_9510615�
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
E__inference_dense_59_layer_call_and_return_conditional_losses_9510626�
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
*:( 2batch_normalization_33/gamma
):' 2batch_normalization_33/beta
2:0  (2"batch_normalization_33/moving_mean
6:4  (2&batch_normalization_33/moving_variance
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
8__inference_batch_normalization_33_layer_call_fn_9510639
8__inference_batch_normalization_33_layer_call_fn_9510652�
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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9510672
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9510706�
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
,__inference_dropout_51_layer_call_fn_9510711
,__inference_dropout_51_layer_call_fn_9510716�
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
G__inference_dropout_51_layer_call_and_return_conditional_losses_9510721
G__inference_dropout_51_layer_call_and_return_conditional_losses_9510733�
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
@__inference_token_and_position_embedding_8_layer_call_fn_9510742�
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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9510767�
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
!:  2dense_60/kernel
: 2dense_60/bias
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
*__inference_dense_60_layer_call_fn_9510776�
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
E__inference_dense_60_layer_call_and_return_conditional_losses_9510787�
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
5__inference_transformer_block_8_layer_call_fn_9510824
5__inference_transformer_block_8_layer_call_fn_9510861�
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
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9510988
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9511128�
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
*:( 2batch_normalization_34/gamma
):' 2batch_normalization_34/beta
2:0  (2"batch_normalization_34/moving_mean
6:4  (2&batch_normalization_34/moving_variance
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
8__inference_batch_normalization_34_layer_call_fn_9511141
8__inference_batch_normalization_34_layer_call_fn_9511154�
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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9511174
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9511208�
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
<__inference_global_average_pooling1d_8_layer_call_fn_9511213�
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
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9511219�
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
,__inference_dropout_52_layer_call_fn_9511224
,__inference_dropout_52_layer_call_fn_9511229�
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
G__inference_dropout_52_layer_call_and_return_conditional_losses_9511234
G__inference_dropout_52_layer_call_and_return_conditional_losses_9511246�
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
!: 2dense_58/kernel
: 2dense_58/bias
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
*__inference_dense_58_layer_call_fn_9511255�
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
E__inference_dense_58_layer_call_and_return_conditional_losses_9511266�
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
!:  2dense_61/kernel
: 2dense_61/bias
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
*__inference_dense_61_layer_call_fn_9511275�
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
E__inference_dense_61_layer_call_and_return_conditional_losses_9511286�
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
*:( 2batch_normalization_32/gamma
):' 2batch_normalization_32/beta
2:0  (2"batch_normalization_32/moving_mean
6:4  (2&batch_normalization_32/moving_variance
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
8__inference_batch_normalization_32_layer_call_fn_9511299
8__inference_batch_normalization_32_layer_call_fn_9511312�
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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9511332
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9511366�
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
*:( 2batch_normalization_35/gamma
):' 2batch_normalization_35/beta
2:0  (2"batch_normalization_35/moving_mean
6:4  (2&batch_normalization_35/moving_variance
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
8__inference_batch_normalization_35_layer_call_fn_9511379
8__inference_batch_normalization_35_layer_call_fn_9511392�
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9511412
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9511446�
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
,__inference_dropout_50_layer_call_fn_9511451
,__inference_dropout_50_layer_call_fn_9511456�
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
G__inference_dropout_50_layer_call_and_return_conditional_losses_9511461
G__inference_dropout_50_layer_call_and_return_conditional_losses_9511473�
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
,__inference_dropout_53_layer_call_fn_9511478
,__inference_dropout_53_layer_call_fn_9511483�
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
G__inference_dropout_53_layer_call_and_return_conditional_losses_9511488
G__inference_dropout_53_layer_call_and_return_conditional_losses_9511500�
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
/__inference_concatenate_8_layer_call_fn_9511506�
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
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9511513�
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
!:@2dense_62/kernel
:2dense_62/bias
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
*__inference_dense_62_layer_call_fn_9511522�
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
E__inference_dense_62_layer_call_and_return_conditional_losses_9511533�
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
H:F26token_and_position_embedding_8/embedding_16/embeddings
H:F	26token_and_position_embedding_8/embedding_17/embeddings
@:>2*transformer_block_8/attention/query/kernel
::82(transformer_block_8/attention/query/bias
>:<2(transformer_block_8/attention/key/kernel
8:62&transformer_block_8/attention/key/bias
@:>2*transformer_block_8/attention/value/kernel
::82(transformer_block_8/attention/value/bias
K:I25transformer_block_8/attention/attention_output/kernel
A:?23transformer_block_8/attention/attention_output/bias
!:2dense_56/kernel
:2dense_56/bias
!:2dense_57/kernel
:2dense_57/bias
>:<20transformer_block_8/layer_normalization_16/gamma
=:;2/transformer_block_8/layer_normalization_16/beta
>:<20transformer_block_8/layer_normalization_17/gamma
=:;2/transformer_block_8/layer_normalization_17/beta
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
%__inference_signature_wrapper_9510606input_17input_18"�
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
.__inference_sequential_8_layer_call_fn_9507909
.__inference_sequential_8_layer_call_fn_9511546
.__inference_sequential_8_layer_call_fn_9511559
.__inference_sequential_8_layer_call_fn_9507982�
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_9511616
I__inference_sequential_8_layer_call_and_return_conditional_losses_9511673
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507996
I__inference_sequential_8_layer_call_and_return_conditional_losses_9508010�
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
*__inference_dense_56_layer_call_fn_9511682�
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
E__inference_dense_56_layer_call_and_return_conditional_losses_9511713�
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
*__inference_dense_57_layer_call_fn_9511722�
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
E__inference_dense_57_layer_call_and_return_conditional_losses_9511752�
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
 2Adam/dense_59/kernel/m
 : 2Adam/dense_59/bias/m
/:- 2#Adam/batch_normalization_33/gamma/m
.:, 2"Adam/batch_normalization_33/beta/m
&:$  2Adam/dense_60/kernel/m
 : 2Adam/dense_60/bias/m
/:- 2#Adam/batch_normalization_34/gamma/m
.:, 2"Adam/batch_normalization_34/beta/m
&:$ 2Adam/dense_58/kernel/m
 : 2Adam/dense_58/bias/m
&:$  2Adam/dense_61/kernel/m
 : 2Adam/dense_61/bias/m
/:- 2#Adam/batch_normalization_32/gamma/m
.:, 2"Adam/batch_normalization_32/beta/m
/:- 2#Adam/batch_normalization_35/gamma/m
.:, 2"Adam/batch_normalization_35/beta/m
&:$@2Adam/dense_62/kernel/m
 :2Adam/dense_62/bias/m
M:K2=Adam/token_and_position_embedding_8/embedding_16/embeddings/m
M:K	2=Adam/token_and_position_embedding_8/embedding_17/embeddings/m
E:C21Adam/transformer_block_8/attention/query/kernel/m
?:=2/Adam/transformer_block_8/attention/query/bias/m
C:A2/Adam/transformer_block_8/attention/key/kernel/m
=:;2-Adam/transformer_block_8/attention/key/bias/m
E:C21Adam/transformer_block_8/attention/value/kernel/m
?:=2/Adam/transformer_block_8/attention/value/bias/m
P:N2<Adam/transformer_block_8/attention/attention_output/kernel/m
F:D2:Adam/transformer_block_8/attention/attention_output/bias/m
&:$2Adam/dense_56/kernel/m
 :2Adam/dense_56/bias/m
&:$2Adam/dense_57/kernel/m
 :2Adam/dense_57/bias/m
C:A27Adam/transformer_block_8/layer_normalization_16/gamma/m
B:@26Adam/transformer_block_8/layer_normalization_16/beta/m
C:A27Adam/transformer_block_8/layer_normalization_17/gamma/m
B:@26Adam/transformer_block_8/layer_normalization_17/beta/m
&:$
 2Adam/dense_59/kernel/v
 : 2Adam/dense_59/bias/v
/:- 2#Adam/batch_normalization_33/gamma/v
.:, 2"Adam/batch_normalization_33/beta/v
&:$  2Adam/dense_60/kernel/v
 : 2Adam/dense_60/bias/v
/:- 2#Adam/batch_normalization_34/gamma/v
.:, 2"Adam/batch_normalization_34/beta/v
&:$ 2Adam/dense_58/kernel/v
 : 2Adam/dense_58/bias/v
&:$  2Adam/dense_61/kernel/v
 : 2Adam/dense_61/bias/v
/:- 2#Adam/batch_normalization_32/gamma/v
.:, 2"Adam/batch_normalization_32/beta/v
/:- 2#Adam/batch_normalization_35/gamma/v
.:, 2"Adam/batch_normalization_35/beta/v
&:$@2Adam/dense_62/kernel/v
 :2Adam/dense_62/bias/v
M:K2=Adam/token_and_position_embedding_8/embedding_16/embeddings/v
M:K	2=Adam/token_and_position_embedding_8/embedding_17/embeddings/v
E:C21Adam/transformer_block_8/attention/query/kernel/v
?:=2/Adam/transformer_block_8/attention/query/bias/v
C:A2/Adam/transformer_block_8/attention/key/kernel/v
=:;2-Adam/transformer_block_8/attention/key/bias/v
E:C21Adam/transformer_block_8/attention/value/kernel/v
?:=2/Adam/transformer_block_8/attention/value/bias/v
P:N2<Adam/transformer_block_8/attention/attention_output/kernel/v
F:D2:Adam/transformer_block_8/attention/attention_output/bias/v
&:$2Adam/dense_56/kernel/v
 :2Adam/dense_56/bias/v
&:$2Adam/dense_57/kernel/v
 :2Adam/dense_57/bias/v
C:A27Adam/transformer_block_8/layer_normalization_16/gamma/v
B:@26Adam/transformer_block_8/layer_normalization_16/beta/v
C:A27Adam/transformer_block_8/layer_normalization_17/gamma/v
B:@26Adam/transformer_block_8/layer_normalization_17/beta/v�
"__inference__wrapped_model_9507735�D)&('?@��WTVU����������������stkl����|~}��Z�W
P�M
K�H
"�
input_17���������	
"�
input_18���������

� "3�0
.
dense_62"�
dense_62����������
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9511332b|~}3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_9511366b~|}3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
8__inference_batch_normalization_32_layer_call_fn_9511299U|~}3�0
)�&
 �
inputs��������� 
p 
� "���������� �
8__inference_batch_normalization_32_layer_call_fn_9511312U~|}3�0
)�&
 �
inputs��������� 
p
� "���������� �
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9510672b)&('3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_9510706b()&'3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
8__inference_batch_normalization_33_layer_call_fn_9510639U)&('3�0
)�&
 �
inputs��������� 
p 
� "���������� �
8__inference_batch_normalization_33_layer_call_fn_9510652U()&'3�0
)�&
 �
inputs��������� 
p
� "���������� �
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9511174bWTVU3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_9511208bVWTU3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
8__inference_batch_normalization_34_layer_call_fn_9511141UWTVU3�0
)�&
 �
inputs��������� 
p 
� "���������� �
8__inference_batch_normalization_34_layer_call_fn_9511154UVWTU3�0
)�&
 �
inputs��������� 
p
� "���������� �
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9511412f����3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_9511446f����3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
8__inference_batch_normalization_35_layer_call_fn_9511379Y����3�0
)�&
 �
inputs��������� 
p 
� "���������� �
8__inference_batch_normalization_35_layer_call_fn_9511392Y����3�0
)�&
 �
inputs��������� 
p
� "���������� �
J__inference_concatenate_8_layer_call_and_return_conditional_losses_9511513�Z�W
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
/__inference_concatenate_8_layer_call_fn_9511506vZ�W
P�M
K�H
"�
inputs/0��������� 
"�
inputs/1��������� 
� "����������@�
E__inference_dense_56_layer_call_and_return_conditional_losses_9511713f��3�0
)�&
$�!
inputs���������	
� ")�&
�
0���������	
� �
*__inference_dense_56_layer_call_fn_9511682Y��3�0
)�&
$�!
inputs���������	
� "����������	�
E__inference_dense_57_layer_call_and_return_conditional_losses_9511752f��3�0
)�&
$�!
inputs���������	
� ")�&
�
0���������	
� �
*__inference_dense_57_layer_call_fn_9511722Y��3�0
)�&
$�!
inputs���������	
� "����������	�
E__inference_dense_58_layer_call_and_return_conditional_losses_9511266\kl/�,
%�"
 �
inputs���������
� "%�"
�
0��������� 
� }
*__inference_dense_58_layer_call_fn_9511255Okl/�,
%�"
 �
inputs���������
� "���������� �
E__inference_dense_59_layer_call_and_return_conditional_losses_9510626\/�,
%�"
 �
inputs���������

� "%�"
�
0��������� 
� }
*__inference_dense_59_layer_call_fn_9510615O/�,
%�"
 �
inputs���������

� "���������� �
E__inference_dense_60_layer_call_and_return_conditional_losses_9510787\?@/�,
%�"
 �
inputs��������� 
� "%�"
�
0��������� 
� }
*__inference_dense_60_layer_call_fn_9510776O?@/�,
%�"
 �
inputs��������� 
� "���������� �
E__inference_dense_61_layer_call_and_return_conditional_losses_9511286\st/�,
%�"
 �
inputs��������� 
� "%�"
�
0��������� 
� }
*__inference_dense_61_layer_call_fn_9511275Ost/�,
%�"
 �
inputs��������� 
� "���������� �
E__inference_dense_62_layer_call_and_return_conditional_losses_9511533^��/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� 
*__inference_dense_62_layer_call_fn_9511522Q��/�,
%�"
 �
inputs���������@
� "�����������
G__inference_dropout_50_layer_call_and_return_conditional_losses_9511461\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
G__inference_dropout_50_layer_call_and_return_conditional_losses_9511473\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� 
,__inference_dropout_50_layer_call_fn_9511451O3�0
)�&
 �
inputs��������� 
p 
� "���������� 
,__inference_dropout_50_layer_call_fn_9511456O3�0
)�&
 �
inputs��������� 
p
� "���������� �
G__inference_dropout_51_layer_call_and_return_conditional_losses_9510721\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
G__inference_dropout_51_layer_call_and_return_conditional_losses_9510733\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� 
,__inference_dropout_51_layer_call_fn_9510711O3�0
)�&
 �
inputs��������� 
p 
� "���������� 
,__inference_dropout_51_layer_call_fn_9510716O3�0
)�&
 �
inputs��������� 
p
� "���������� �
G__inference_dropout_52_layer_call_and_return_conditional_losses_9511234\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
G__inference_dropout_52_layer_call_and_return_conditional_losses_9511246\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� 
,__inference_dropout_52_layer_call_fn_9511224O3�0
)�&
 �
inputs��������� 
p 
� "���������� 
,__inference_dropout_52_layer_call_fn_9511229O3�0
)�&
 �
inputs��������� 
p
� "���������� �
G__inference_dropout_53_layer_call_and_return_conditional_losses_9511488\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
G__inference_dropout_53_layer_call_and_return_conditional_losses_9511500\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� 
,__inference_dropout_53_layer_call_fn_9511478O3�0
)�&
 �
inputs��������� 
p 
� "���������� 
,__inference_dropout_53_layer_call_fn_9511483O3�0
)�&
 �
inputs��������� 
p
� "���������� �
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_9511219{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
<__inference_global_average_pooling1d_8_layer_call_fn_9511213nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
D__inference_model_8_layer_call_and_return_conditional_losses_9509597�D)&('?@��WTVU����������������stkl����|~}��b�_
X�U
K�H
"�
input_17���������	
"�
input_18���������

p 

 
� "%�"
�
0���������
� �
D__inference_model_8_layer_call_and_return_conditional_losses_9509707�D()&'?@��VWTU����������������stkl����~|}��b�_
X�U
K�H
"�
input_17���������	
"�
input_18���������

p

 
� "%�"
�
0���������
� �
D__inference_model_8_layer_call_and_return_conditional_losses_9510157�D)&('?@��WTVU����������������stkl����|~}��b�_
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
D__inference_model_8_layer_call_and_return_conditional_losses_9510510�D()&'?@��VWTU����������������stkl����~|}��b�_
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
)__inference_model_8_layer_call_fn_9508721�D)&('?@��WTVU����������������stkl����|~}��b�_
X�U
K�H
"�
input_17���������	
"�
input_18���������

p 

 
� "�����������
)__inference_model_8_layer_call_fn_9509487�D()&'?@��VWTU����������������stkl����~|}��b�_
X�U
K�H
"�
input_17���������	
"�
input_18���������

p

 
� "�����������
)__inference_model_8_layer_call_fn_9509807�D)&('?@��WTVU����������������stkl����|~}��b�_
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
)__inference_model_8_layer_call_fn_9509901�D()&'?@��VWTU����������������stkl����~|}��b�_
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_9507996z����C�@
9�6
,�)
dense_56_input���������	
p 

 
� ")�&
�
0���������	
� �
I__inference_sequential_8_layer_call_and_return_conditional_losses_9508010z����C�@
9�6
,�)
dense_56_input���������	
p

 
� ")�&
�
0���������	
� �
I__inference_sequential_8_layer_call_and_return_conditional_losses_9511616r����;�8
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_9511673r����;�8
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
.__inference_sequential_8_layer_call_fn_9507909m����C�@
9�6
,�)
dense_56_input���������	
p 

 
� "����������	�
.__inference_sequential_8_layer_call_fn_9507982m����C�@
9�6
,�)
dense_56_input���������	
p

 
� "����������	�
.__inference_sequential_8_layer_call_fn_9511546e����;�8
1�.
$�!
inputs���������	
p 

 
� "����������	�
.__inference_sequential_8_layer_call_fn_9511559e����;�8
1�.
$�!
inputs���������	
p

 
� "����������	�
%__inference_signature_wrapper_9510606�D)&('?@��WTVU����������������stkl����|~}��m�j
� 
c�`
.
input_17"�
input_17���������	
.
input_18"�
input_18���������
"3�0
.
dense_62"�
dense_62����������
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_9510767]��*�'
 �
�
x���������	
� ")�&
�
0���������	
� �
@__inference_token_and_position_embedding_8_layer_call_fn_9510742P��*�'
 �
�
x���������	
� "����������	�
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9510988� ����������������7�4
-�*
$�!
inputs���������	
p 
� ")�&
�
0���������	
� �
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_9511128� ����������������7�4
-�*
$�!
inputs���������	
p
� ")�&
�
0���������	
� �
5__inference_transformer_block_8_layer_call_fn_9510824y ����������������7�4
-�*
$�!
inputs���������	
p 
� "����������	�
5__inference_transformer_block_8_layer_call_fn_9510861y ����������������7�4
-�*
$�!
inputs���������	
p
� "����������	