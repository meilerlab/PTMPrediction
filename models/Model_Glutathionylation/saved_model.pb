е▌
ц╗
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Н
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
е
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.12v2.8.0-80-g0516d4d8bce8ре
И
embedding_8/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_8/embeddings
Б
*embedding_8/embeddings/Read/ReadVariableOpReadVariableOpembedding_8/embeddings*
_output_shapes

:*
dtype0
Р
batch_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_32/gamma
Й
0batch_normalization_32/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_32/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_32/beta
З
/batch_normalization_32/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_32/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_32/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_32/moving_mean
Х
6batch_normalization_32/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_32/moving_mean*
_output_shapes
:*
dtype0
д
&batch_normalization_32/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_32/moving_variance
Э
:batch_normalization_32/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_32/moving_variance*
_output_shapes
:*
dtype0
z
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_33/kernel
s
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes

:
*
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:*
dtype0
Р
batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_34/gamma
Й
0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_34/beta
З
/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_34/moving_mean
Х
6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes
:*
dtype0
д
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_34/moving_variance
Э
:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes
:*
dtype0
z
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

:*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:*
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

:*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:*
dtype0
Р
batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_33/gamma
Й
0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_33/beta
З
/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_33/moving_mean
Х
6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes
:*
dtype0
д
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_33/moving_variance
Э
:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes
:*
dtype0
Р
batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_35/gamma
Й
0batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_35/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_35/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_35/beta
З
/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_35/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_35/moving_mean
Х
6batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_35/moving_mean*
_output_shapes
:*
dtype0
д
&batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_35/moving_variance
Э
:batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_35/moving_variance*
_output_shapes
:*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:*
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
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
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
Ц
Adam/embedding_8/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_8/embeddings/m
П
1Adam/embedding_8/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_8/embeddings/m*
_output_shapes

:*
dtype0
Ю
#Adam/batch_normalization_32/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_32/gamma/m
Ч
7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_32/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_32/beta/m
Х
6Adam/batch_normalization_32/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/m*
_output_shapes
:*
dtype0
И
Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_33/kernel/m
Б
*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m*
_output_shapes

:
*
dtype0
А
Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/m
y
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_34/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_34/gamma/m
Ч
7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_34/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_34/beta/m
Х
6Adam/batch_normalization_34/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/m*
_output_shapes
:*
dtype0
И
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_32/kernel/m
Б
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/m
y
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_34/kernel/m
Б
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_34/bias/m
y
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_33/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_33/gamma/m
Ч
7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_33/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_33/beta/m
Х
6Adam/batch_normalization_33/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_35/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_35/gamma/m
Ч
7Adam/batch_normalization_35/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_35/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_35/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_35/beta/m
Х
6Adam/batch_normalization_35/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_35/beta/m*
_output_shapes
:*
dtype0
И
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_35/kernel/m
Б
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:*
dtype0
Ц
Adam/embedding_8/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_8/embeddings/v
П
1Adam/embedding_8/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_8/embeddings/v*
_output_shapes

:*
dtype0
Ю
#Adam/batch_normalization_32/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_32/gamma/v
Ч
7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_32/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_32/beta/v
Х
6Adam/batch_normalization_32/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/v*
_output_shapes
:*
dtype0
И
Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_33/kernel/v
Б
*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v*
_output_shapes

:
*
dtype0
А
Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/v
y
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_34/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_34/gamma/v
Ч
7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_34/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_34/beta/v
Х
6Adam/batch_normalization_34/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/v*
_output_shapes
:*
dtype0
И
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_32/kernel/v
Б
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/v
y
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_34/kernel/v
Б
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_34/bias/v
y
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_33/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_33/gamma/v
Ч
7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_33/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_33/beta/v
Х
6Adam/batch_normalization_33/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_35/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_35/gamma/v
Ч
7Adam/batch_normalization_35/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_35/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_35/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_35/beta/v
Х
6Adam/batch_normalization_35/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_35/beta/v*
_output_shapes
:*
dtype0
И
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_35/kernel/v
Б
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
СО
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╦Н
value└НB╝Н B┤Н
Ъ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
а

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
* 
╒
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
ж

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*
е
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9_random_generator
:__call__
*;&call_and_return_all_conditional_losses* 
╒
<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
О
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
е
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses* 
ж

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses*
ж

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses*
╒
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
╒
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses*
ж
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~_random_generator
__call__
+А&call_and_return_all_conditional_losses* 
м
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е_random_generator
Ж__call__
+З&call_and_return_all_conditional_losses* 
Ф
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses* 
о
Оkernel
	Пbias
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses*
б
	Цiter
Чbeta_1
Шbeta_2

Щdecay
Ъlearning_ratemё#mЄ$mє-mЇ.mї=mЎ>mўTm°Um∙\m·]m√em№fm¤pm■qm 	ОmА	ПmБvВ#vГ$vД-vЕ.vЖ=vЗ>vИTvЙUvК\vЛ]vМevНfvОpvПqvР	ОvС	ПvТ*
─
0
#1
$2
%3
&4
-5
.6
=7
>8
?9
@10
T11
U12
\13
]14
e15
f16
g17
h18
p19
q20
r21
s22
О23
П24*
Д
0
#1
$2
-3
.4
=5
>6
T7
U8
\9
]10
e11
f12
p13
q14
О15
П16*
* 
╡
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

аserving_default* 
jd
VARIABLE_VALUEembedding_8/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
Ш
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_32/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_32/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_32/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_32/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
#0
$1
%2
&3*

#0
$1*
* 
Ш
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_33/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 
Ш
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
5	variables
6trainable_variables
7regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_34/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_34/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_34/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_34/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
=0
>1
?2
@3*

=0
>1*
* 
Ш
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ц
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_32/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

T0
U1*

T0
U1*
* 
Ш
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_34/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

\0
]1*

\0
]1*
* 
Ш
╔non_trainable_variables
╩layers
╦metrics
 ╠layer_regularization_losses
═layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_33/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_33/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_33/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_33/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
e0
f1
g2
h3*

e0
f1*
* 
Ш
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_35/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_35/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_35/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_35/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
p0
q1
r2
s3*

p0
q1*
* 
Ш
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ш
╪non_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
z	variables
{trainable_variables
|regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_35/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

О0
П1*

О0
П1*
* 
Ю
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
<
%0
&1
?2
@3
g4
h5
r6
s7*
В
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
16*

ь0*
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1*
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
?0
@1*
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
g0
h1*
* 
* 
* 
* 

r0
s1*
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

эtotal

юcount
я	variables
Ё	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

э0
ю1*

я	variables*
ОЗ
VARIABLE_VALUEAdam/embedding_8/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_32/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_33/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_33/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_34/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_34/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_34/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_34/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_33/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_35/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_35/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUEAdam/embedding_8/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_32/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_33/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_33/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_34/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_34/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_34/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_34/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_33/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_35/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_35/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
serving_default_input_17Placeholder*'
_output_shapes
:         	*
dtype0*
shape:         	
{
serving_default_input_18Placeholder*'
_output_shapes
:         
*
dtype0*
shape:         

┤
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17serving_default_input_18embedding_8/embeddingsdense_33/kerneldense_33/bias&batch_normalization_32/moving_variancebatch_normalization_32/gamma"batch_normalization_32/moving_meanbatch_normalization_32/beta&batch_normalization_34/moving_variancebatch_normalization_34/gamma"batch_normalization_34/moving_meanbatch_normalization_34/betadense_34/kerneldense_34/biasdense_32/kerneldense_32/bias&batch_normalization_35/moving_variancebatch_normalization_35/gamma"batch_normalization_35/moving_meanbatch_normalization_35/beta&batch_normalization_33/moving_variancebatch_normalization_33/gamma"batch_normalization_33/moving_meanbatch_normalization_33/betadense_35/kerneldense_35/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_921543
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
В
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_8/embeddings/Read/ReadVariableOp0batch_normalization_32/gamma/Read/ReadVariableOp/batch_normalization_32/beta/Read/ReadVariableOp6batch_normalization_32/moving_mean/Read/ReadVariableOp:batch_normalization_32/moving_variance/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp0batch_normalization_34/gamma/Read/ReadVariableOp/batch_normalization_34/beta/Read/ReadVariableOp6batch_normalization_34/moving_mean/Read/ReadVariableOp:batch_normalization_34/moving_variance/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp0batch_normalization_33/gamma/Read/ReadVariableOp/batch_normalization_33/beta/Read/ReadVariableOp6batch_normalization_33/moving_mean/Read/ReadVariableOp:batch_normalization_33/moving_variance/Read/ReadVariableOp0batch_normalization_35/gamma/Read/ReadVariableOp/batch_normalization_35/beta/Read/ReadVariableOp6batch_normalization_35/moving_mean/Read/ReadVariableOp:batch_normalization_35/moving_variance/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_8/embeddings/m/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_32/beta/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_34/beta/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_33/beta/m/Read/ReadVariableOp7Adam/batch_normalization_35/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_35/beta/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp1Adam/embedding_8/embeddings/v/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_32/beta/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_34/beta/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_33/beta/v/Read/ReadVariableOp7Adam/batch_normalization_35/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_35/beta/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpConst*O
TinH
F2D	*
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
GPU 2J 8В *(
f#R!
__inference__traced_save_922314
╒
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_8/embeddingsbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_variancedense_33/kerneldense_33/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_variancedense_32/kerneldense_32/biasdense_34/kerneldense_34/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_variancebatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/embedding_8/embeddings/m#Adam/batch_normalization_32/gamma/m"Adam/batch_normalization_32/beta/mAdam/dense_33/kernel/mAdam/dense_33/bias/m#Adam/batch_normalization_34/gamma/m"Adam/batch_normalization_34/beta/mAdam/dense_32/kernel/mAdam/dense_32/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/m#Adam/batch_normalization_33/gamma/m"Adam/batch_normalization_33/beta/m#Adam/batch_normalization_35/gamma/m"Adam/batch_normalization_35/beta/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/embedding_8/embeddings/v#Adam/batch_normalization_32/gamma/v"Adam/batch_normalization_32/beta/vAdam/dense_33/kernel/vAdam/dense_33/bias/v#Adam/batch_normalization_34/gamma/v"Adam/batch_normalization_34/beta/vAdam/dense_32/kernel/vAdam/dense_32/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/v#Adam/batch_normalization_33/gamma/v"Adam/batch_normalization_33/beta/v#Adam/batch_normalization_35/gamma/v"Adam/batch_normalization_35/beta/vAdam/dense_35/kernel/vAdam/dense_35/bias/v*N
TinG
E2C*
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_922522╖╥
у
■
$__inference_signature_wrapper_921543
input_17
input_18
unknown:
	unknown_0:

	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identityИвStatefulPartitionedCall№
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
unknown_23*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_919976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
input_17:QM
'
_output_shapes
:         

"
_user_specified_name
input_18
╖P
·
C__inference_model_8_layer_call_and_return_conditional_losses_921061
input_17
input_18$
embedding_8_920995:!
dense_33_920998:

dense_33_921000:+
batch_normalization_32_921003:+
batch_normalization_32_921005:+
batch_normalization_32_921007:+
batch_normalization_32_921009:+
batch_normalization_34_921012:+
batch_normalization_34_921014:+
batch_normalization_34_921016:+
batch_normalization_34_921018:!
dense_34_921024:
dense_34_921026:!
dense_32_921029:
dense_32_921031:+
batch_normalization_35_921034:+
batch_normalization_35_921036:+
batch_normalization_35_921038:+
batch_normalization_35_921040:+
batch_normalization_33_921043:+
batch_normalization_33_921045:+
batch_normalization_33_921047:+
batch_normalization_33_921049:!
dense_35_921055:
dense_35_921057:
identityИв.batch_normalization_32/StatefulPartitionedCallв.batch_normalization_33/StatefulPartitionedCallв.batch_normalization_34/StatefulPartitionedCallв.batch_normalization_35/StatefulPartitionedCallв dense_32/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв dense_34/StatefulPartitionedCallв dense_35/StatefulPartitionedCallв"dropout_32/StatefulPartitionedCallв"dropout_33/StatefulPartitionedCallв"dropout_34/StatefulPartitionedCallв"dropout_35/StatefulPartitionedCallв#embedding_8/StatefulPartitionedCallь
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_17embedding_8_920995*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_920336Є
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinput_18dense_33_920998dense_33_921000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_920351Т
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall,embedding_8/StatefulPartitionedCall:output:0batch_normalization_32_921003batch_normalization_32_921005batch_normalization_32_921007batch_normalization_32_921009*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_920047Л
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0batch_normalization_34_921012batch_normalization_34_921014batch_normalization_34_921016batch_normalization_34_921018*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_920129Б
"dropout_32/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_920662в
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0#^dropout_32/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_920639Б
*global_average_pooling1d_8/PartitionedCallPartitionedCall+dropout_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_920150Х
 dense_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0dense_34_921024dense_34_921026*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_920401Э
 dense_32/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_32_921029dense_32_921031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_920418Л
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0batch_normalization_35_921034batch_normalization_35_921036batch_normalization_35_921038batch_normalization_35_921040*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_920306Л
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0batch_normalization_33_921043batch_normalization_33_921045batch_normalization_33_921047batch_normalization_33_921049*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_920224в
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0#^dropout_34/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_920596в
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0#^dropout_33/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_920573Х
concatenate_8/PartitionedCallPartitionedCall+dropout_33/StatefulPartitionedCall:output:0+dropout_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_concatenate_8_layer_call_and_return_conditional_losses_920463Р
 dense_35/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_35_921055dense_35_921057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_920476x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╨
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall#^dropout_32/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2H
"dropout_32/StatefulPartitionedCall"dropout_32/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
input_17:QM
'
_output_shapes
:         

"
_user_specified_name
input_18
б
G
+__inference_dropout_33_layer_call_fn_922010

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_920447`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы

ї
D__inference_dense_34_layer_call_and_return_conditional_losses_920401

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
пP
°
C__inference_model_8_layer_call_and_return_conditional_losses_920812

inputs
inputs_1$
embedding_8_920746:!
dense_33_920749:

dense_33_920751:+
batch_normalization_32_920754:+
batch_normalization_32_920756:+
batch_normalization_32_920758:+
batch_normalization_32_920760:+
batch_normalization_34_920763:+
batch_normalization_34_920765:+
batch_normalization_34_920767:+
batch_normalization_34_920769:!
dense_34_920775:
dense_34_920777:!
dense_32_920780:
dense_32_920782:+
batch_normalization_35_920785:+
batch_normalization_35_920787:+
batch_normalization_35_920789:+
batch_normalization_35_920791:+
batch_normalization_33_920794:+
batch_normalization_33_920796:+
batch_normalization_33_920798:+
batch_normalization_33_920800:!
dense_35_920806:
dense_35_920808:
identityИв.batch_normalization_32/StatefulPartitionedCallв.batch_normalization_33/StatefulPartitionedCallв.batch_normalization_34/StatefulPartitionedCallв.batch_normalization_35/StatefulPartitionedCallв dense_32/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв dense_34/StatefulPartitionedCallв dense_35/StatefulPartitionedCallв"dropout_32/StatefulPartitionedCallв"dropout_33/StatefulPartitionedCallв"dropout_34/StatefulPartitionedCallв"dropout_35/StatefulPartitionedCallв#embedding_8/StatefulPartitionedCallъ
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_8_920746*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_920336Є
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_33_920749dense_33_920751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_920351Т
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall,embedding_8/StatefulPartitionedCall:output:0batch_normalization_32_920754batch_normalization_32_920756batch_normalization_32_920758batch_normalization_32_920760*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_920047Л
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0batch_normalization_34_920763batch_normalization_34_920765batch_normalization_34_920767batch_normalization_34_920769*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_920129Б
"dropout_32/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_920662в
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0#^dropout_32/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_920639Б
*global_average_pooling1d_8/PartitionedCallPartitionedCall+dropout_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_920150Х
 dense_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0dense_34_920775dense_34_920777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_920401Э
 dense_32/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_32_920780dense_32_920782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_920418Л
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0batch_normalization_35_920785batch_normalization_35_920787batch_normalization_35_920789batch_normalization_35_920791*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_920306Л
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0batch_normalization_33_920794batch_normalization_33_920796batch_normalization_33_920798batch_normalization_33_920800*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_920224в
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0#^dropout_34/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_920596в
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0#^dropout_33/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_920573Х
concatenate_8/PartitionedCallPartitionedCall+dropout_33/StatefulPartitionedCall:output:0+dropout_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_concatenate_8_layer_call_and_return_conditional_losses_920463Р
 dense_35/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_35_920806dense_35_920808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_920476x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╨
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall#^dropout_32/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2H
"dropout_32/StatefulPartitionedCall"dropout_32/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         

 
_user_specified_nameinputs
У%
ы
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_920129

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
╥
7__inference_batch_normalization_35_layer_call_fn_921951

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_920306o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
■%
ы
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_920047

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       в
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
║
s
I__inference_concatenate_8_layer_call_and_return_conditional_losses_920463

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
:         W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
┌Я
а
C__inference_model_8_layer_call_and_return_conditional_losses_921485
inputs_0
inputs_15
#embedding_8_embedding_lookup_921295:9
'dense_33_matmul_readvariableop_resource:
6
(dense_33_biasadd_readvariableop_resource:L
>batch_normalization_32_assignmovingavg_readvariableop_resource:N
@batch_normalization_32_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_32_batchnorm_mul_readvariableop_resource:F
8batch_normalization_32_batchnorm_readvariableop_resource:L
>batch_normalization_34_assignmovingavg_readvariableop_resource:N
@batch_normalization_34_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_34_batchnorm_mul_readvariableop_resource:F
8batch_normalization_34_batchnorm_readvariableop_resource:9
'dense_34_matmul_readvariableop_resource:6
(dense_34_biasadd_readvariableop_resource:9
'dense_32_matmul_readvariableop_resource:6
(dense_32_biasadd_readvariableop_resource:L
>batch_normalization_35_assignmovingavg_readvariableop_resource:N
@batch_normalization_35_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_35_batchnorm_mul_readvariableop_resource:F
8batch_normalization_35_batchnorm_readvariableop_resource:L
>batch_normalization_33_assignmovingavg_readvariableop_resource:N
@batch_normalization_33_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_33_batchnorm_mul_readvariableop_resource:F
8batch_normalization_33_batchnorm_readvariableop_resource:9
'dense_35_matmul_readvariableop_resource:6
(dense_35_biasadd_readvariableop_resource:
identityИв&batch_normalization_32/AssignMovingAvgв5batch_normalization_32/AssignMovingAvg/ReadVariableOpв(batch_normalization_32/AssignMovingAvg_1в7batch_normalization_32/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_32/batchnorm/ReadVariableOpв3batch_normalization_32/batchnorm/mul/ReadVariableOpв&batch_normalization_33/AssignMovingAvgв5batch_normalization_33/AssignMovingAvg/ReadVariableOpв(batch_normalization_33/AssignMovingAvg_1в7batch_normalization_33/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_33/batchnorm/ReadVariableOpв3batch_normalization_33/batchnorm/mul/ReadVariableOpв&batch_normalization_34/AssignMovingAvgв5batch_normalization_34/AssignMovingAvg/ReadVariableOpв(batch_normalization_34/AssignMovingAvg_1в7batch_normalization_34/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_34/batchnorm/ReadVariableOpв3batch_normalization_34/batchnorm/mul/ReadVariableOpв&batch_normalization_35/AssignMovingAvgв5batch_normalization_35/AssignMovingAvg/ReadVariableOpв(batch_normalization_35/AssignMovingAvg_1в7batch_normalization_35/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_35/batchnorm/ReadVariableOpв3batch_normalization_35/batchnorm/mul/ReadVariableOpвdense_32/BiasAdd/ReadVariableOpвdense_32/MatMul/ReadVariableOpвdense_33/BiasAdd/ReadVariableOpвdense_33/MatMul/ReadVariableOpвdense_34/BiasAdd/ReadVariableOpвdense_34/MatMul/ReadVariableOpвdense_35/BiasAdd/ReadVariableOpвdense_35/MatMul/ReadVariableOpвembedding_8/embedding_lookupc
embedding_8/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:         	ы
embedding_8/embedding_lookupResourceGather#embedding_8_embedding_lookup_921295embedding_8/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_8/embedding_lookup/921295*+
_output_shapes
:         	*
dtype0╞
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_8/embedding_lookup/921295*+
_output_shapes
:         	Щ
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         	Ж
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0}
dense_33/MatMulMatMulinputs_1&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:         Ж
5batch_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       █
#batch_normalization_32/moments/meanMean0embedding_8/embedding_lookup/Identity_1:output:0>batch_normalization_32/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ц
+batch_normalization_32/moments/StopGradientStopGradient,batch_normalization_32/moments/mean:output:0*
T0*"
_output_shapes
:у
0batch_normalization_32/moments/SquaredDifferenceSquaredDifference0embedding_8/embedding_lookup/Identity_1:output:04batch_normalization_32/moments/StopGradient:output:0*
T0*+
_output_shapes
:         	К
9batch_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ч
'batch_normalization_32/moments/varianceMean4batch_normalization_32/moments/SquaredDifference:z:0Bbatch_normalization_32/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ь
&batch_normalization_32/moments/SqueezeSqueeze,batch_normalization_32/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 в
(batch_normalization_32/moments/Squeeze_1Squeeze0batch_normalization_32/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_32/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<░
5batch_normalization_32/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_32_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0╞
*batch_normalization_32/AssignMovingAvg/subSub=batch_normalization_32/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_32/moments/Squeeze:output:0*
T0*
_output_shapes
:╜
*batch_normalization_32/AssignMovingAvg/mulMul.batch_normalization_32/AssignMovingAvg/sub:z:05batch_normalization_32/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:И
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
╫#<┤
7batch_normalization_32/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_32_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0╠
,batch_normalization_32/AssignMovingAvg_1/subSub?batch_normalization_32/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_32/moments/Squeeze_1:output:0*
T0*
_output_shapes
:├
,batch_normalization_32/AssignMovingAvg_1/mulMul0batch_normalization_32/AssignMovingAvg_1/sub:z:07batch_normalization_32/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Р
(batch_normalization_32/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_32_assignmovingavg_1_readvariableop_resource0batch_normalization_32/AssignMovingAvg_1/mul:z:08^batch_normalization_32/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╢
$batch_normalization_32/batchnorm/addAddV21batch_normalization_32/moments/Squeeze_1:output:0/batch_normalization_32/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_32/batchnorm/RsqrtRsqrt(batch_normalization_32/batchnorm/add:z:0*
T0*
_output_shapes
:м
3batch_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╣
$batch_normalization_32/batchnorm/mulMul*batch_normalization_32/batchnorm/Rsqrt:y:0;batch_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:┐
&batch_normalization_32/batchnorm/mul_1Mul0embedding_8/embedding_lookup/Identity_1:output:0(batch_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:         	н
&batch_normalization_32/batchnorm/mul_2Mul/batch_normalization_32/moments/Squeeze:output:0(batch_normalization_32/batchnorm/mul:z:0*
T0*
_output_shapes
:д
/batch_normalization_32/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╡
$batch_normalization_32/batchnorm/subSub7batch_normalization_32/batchnorm/ReadVariableOp:value:0*batch_normalization_32/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╗
&batch_normalization_32/batchnorm/add_1AddV2*batch_normalization_32/batchnorm/mul_1:z:0(batch_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:         	
5batch_normalization_34/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ┬
#batch_normalization_34/moments/meanMeandense_33/Relu:activations:0>batch_normalization_34/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Т
+batch_normalization_34/moments/StopGradientStopGradient,batch_normalization_34/moments/mean:output:0*
T0*
_output_shapes

:╩
0batch_normalization_34/moments/SquaredDifferenceSquaredDifferencedense_33/Relu:activations:04batch_normalization_34/moments/StopGradient:output:0*
T0*'
_output_shapes
:         Г
9batch_normalization_34/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: у
'batch_normalization_34/moments/varianceMean4batch_normalization_34/moments/SquaredDifference:z:0Bbatch_normalization_34/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Ы
&batch_normalization_34/moments/SqueezeSqueeze,batch_normalization_34/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 б
(batch_normalization_34/moments/Squeeze_1Squeeze0batch_normalization_34/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_34/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<░
5batch_normalization_34/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_34_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0╞
*batch_normalization_34/AssignMovingAvg/subSub=batch_normalization_34/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_34/moments/Squeeze:output:0*
T0*
_output_shapes
:╜
*batch_normalization_34/AssignMovingAvg/mulMul.batch_normalization_34/AssignMovingAvg/sub:z:05batch_normalization_34/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:И
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
╫#<┤
7batch_normalization_34/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_34_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0╠
,batch_normalization_34/AssignMovingAvg_1/subSub?batch_normalization_34/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_34/moments/Squeeze_1:output:0*
T0*
_output_shapes
:├
,batch_normalization_34/AssignMovingAvg_1/mulMul0batch_normalization_34/AssignMovingAvg_1/sub:z:07batch_normalization_34/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Р
(batch_normalization_34/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_34_assignmovingavg_1_readvariableop_resource0batch_normalization_34/AssignMovingAvg_1/mul:z:08^batch_normalization_34/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_34/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╢
$batch_normalization_34/batchnorm/addAddV21batch_normalization_34/moments/Squeeze_1:output:0/batch_normalization_34/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_34/batchnorm/RsqrtRsqrt(batch_normalization_34/batchnorm/add:z:0*
T0*
_output_shapes
:м
3batch_normalization_34/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_34_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╣
$batch_normalization_34/batchnorm/mulMul*batch_normalization_34/batchnorm/Rsqrt:y:0;batch_normalization_34/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ж
&batch_normalization_34/batchnorm/mul_1Muldense_33/Relu:activations:0(batch_normalization_34/batchnorm/mul:z:0*
T0*'
_output_shapes
:         н
&batch_normalization_34/batchnorm/mul_2Mul/batch_normalization_34/moments/Squeeze:output:0(batch_normalization_34/batchnorm/mul:z:0*
T0*
_output_shapes
:д
/batch_normalization_34/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_34_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╡
$batch_normalization_34/batchnorm/subSub7batch_normalization_34/batchnorm/ReadVariableOp:value:0*batch_normalization_34/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╖
&batch_normalization_34/batchnorm/add_1AddV2*batch_normalization_34/batchnorm/mul_1:z:0(batch_normalization_34/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ]
dropout_32/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?в
dropout_32/dropout/MulMul*batch_normalization_32/batchnorm/add_1:z:0!dropout_32/dropout/Const:output:0*
T0*+
_output_shapes
:         	r
dropout_32/dropout/ShapeShape*batch_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:ж
/dropout_32/dropout/random_uniform/RandomUniformRandomUniform!dropout_32/dropout/Shape:output:0*
T0*+
_output_shapes
:         	*
dtype0f
!dropout_32/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╦
dropout_32/dropout/GreaterEqualGreaterEqual8dropout_32/dropout/random_uniform/RandomUniform:output:0*dropout_32/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         	Й
dropout_32/dropout/CastCast#dropout_32/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         	О
dropout_32/dropout/Mul_1Muldropout_32/dropout/Mul:z:0dropout_32/dropout/Cast:y:0*
T0*+
_output_shapes
:         	]
dropout_34/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?Ю
dropout_34/dropout/MulMul*batch_normalization_34/batchnorm/add_1:z:0!dropout_34/dropout/Const:output:0*
T0*'
_output_shapes
:         r
dropout_34/dropout/ShapeShape*batch_normalization_34/batchnorm/add_1:z:0*
T0*
_output_shapes
:в
/dropout_34/dropout/random_uniform/RandomUniformRandomUniform!dropout_34/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0f
!dropout_34/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╟
dropout_34/dropout/GreaterEqualGreaterEqual8dropout_34/dropout/random_uniform/RandomUniform:output:0*dropout_34/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
dropout_34/dropout/CastCast#dropout_34/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         К
dropout_34/dropout/Mul_1Muldropout_34/dropout/Mul:z:0dropout_34/dropout/Cast:y:0*
T0*'
_output_shapes
:         s
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :│
global_average_pooling1d_8/MeanMeandropout_32/dropout/Mul_1:z:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Ж
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:*
dtype0С
dense_34/MatMulMatMuldropout_34/dropout/Mul_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:         Ж
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Э
dense_32/MatMulMatMul(global_average_pooling1d_8/Mean:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:         
5batch_normalization_35/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ┬
#batch_normalization_35/moments/meanMeandense_34/Relu:activations:0>batch_normalization_35/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Т
+batch_normalization_35/moments/StopGradientStopGradient,batch_normalization_35/moments/mean:output:0*
T0*
_output_shapes

:╩
0batch_normalization_35/moments/SquaredDifferenceSquaredDifferencedense_34/Relu:activations:04batch_normalization_35/moments/StopGradient:output:0*
T0*'
_output_shapes
:         Г
9batch_normalization_35/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: у
'batch_normalization_35/moments/varianceMean4batch_normalization_35/moments/SquaredDifference:z:0Bbatch_normalization_35/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Ы
&batch_normalization_35/moments/SqueezeSqueeze,batch_normalization_35/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 б
(batch_normalization_35/moments/Squeeze_1Squeeze0batch_normalization_35/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_35/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<░
5batch_normalization_35/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_35_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0╞
*batch_normalization_35/AssignMovingAvg/subSub=batch_normalization_35/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_35/moments/Squeeze:output:0*
T0*
_output_shapes
:╜
*batch_normalization_35/AssignMovingAvg/mulMul.batch_normalization_35/AssignMovingAvg/sub:z:05batch_normalization_35/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:И
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
╫#<┤
7batch_normalization_35/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_35_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0╠
,batch_normalization_35/AssignMovingAvg_1/subSub?batch_normalization_35/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_35/moments/Squeeze_1:output:0*
T0*
_output_shapes
:├
,batch_normalization_35/AssignMovingAvg_1/mulMul0batch_normalization_35/AssignMovingAvg_1/sub:z:07batch_normalization_35/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Р
(batch_normalization_35/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_35_assignmovingavg_1_readvariableop_resource0batch_normalization_35/AssignMovingAvg_1/mul:z:08^batch_normalization_35/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_35/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╢
$batch_normalization_35/batchnorm/addAddV21batch_normalization_35/moments/Squeeze_1:output:0/batch_normalization_35/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_35/batchnorm/RsqrtRsqrt(batch_normalization_35/batchnorm/add:z:0*
T0*
_output_shapes
:м
3batch_normalization_35/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_35_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╣
$batch_normalization_35/batchnorm/mulMul*batch_normalization_35/batchnorm/Rsqrt:y:0;batch_normalization_35/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ж
&batch_normalization_35/batchnorm/mul_1Muldense_34/Relu:activations:0(batch_normalization_35/batchnorm/mul:z:0*
T0*'
_output_shapes
:         н
&batch_normalization_35/batchnorm/mul_2Mul/batch_normalization_35/moments/Squeeze:output:0(batch_normalization_35/batchnorm/mul:z:0*
T0*
_output_shapes
:д
/batch_normalization_35/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_35_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╡
$batch_normalization_35/batchnorm/subSub7batch_normalization_35/batchnorm/ReadVariableOp:value:0*batch_normalization_35/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╖
&batch_normalization_35/batchnorm/add_1AddV2*batch_normalization_35/batchnorm/mul_1:z:0(batch_normalization_35/batchnorm/sub:z:0*
T0*'
_output_shapes
:         
5batch_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ┬
#batch_normalization_33/moments/meanMeandense_32/Relu:activations:0>batch_normalization_33/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Т
+batch_normalization_33/moments/StopGradientStopGradient,batch_normalization_33/moments/mean:output:0*
T0*
_output_shapes

:╩
0batch_normalization_33/moments/SquaredDifferenceSquaredDifferencedense_32/Relu:activations:04batch_normalization_33/moments/StopGradient:output:0*
T0*'
_output_shapes
:         Г
9batch_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: у
'batch_normalization_33/moments/varianceMean4batch_normalization_33/moments/SquaredDifference:z:0Bbatch_normalization_33/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Ы
&batch_normalization_33/moments/SqueezeSqueeze,batch_normalization_33/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 б
(batch_normalization_33/moments/Squeeze_1Squeeze0batch_normalization_33/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_33/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<░
5batch_normalization_33/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_33_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0╞
*batch_normalization_33/AssignMovingAvg/subSub=batch_normalization_33/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_33/moments/Squeeze:output:0*
T0*
_output_shapes
:╜
*batch_normalization_33/AssignMovingAvg/mulMul.batch_normalization_33/AssignMovingAvg/sub:z:05batch_normalization_33/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:И
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
╫#<┤
7batch_normalization_33/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_33_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0╠
,batch_normalization_33/AssignMovingAvg_1/subSub?batch_normalization_33/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_33/moments/Squeeze_1:output:0*
T0*
_output_shapes
:├
,batch_normalization_33/AssignMovingAvg_1/mulMul0batch_normalization_33/AssignMovingAvg_1/sub:z:07batch_normalization_33/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Р
(batch_normalization_33/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_33_assignmovingavg_1_readvariableop_resource0batch_normalization_33/AssignMovingAvg_1/mul:z:08^batch_normalization_33/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╢
$batch_normalization_33/batchnorm/addAddV21batch_normalization_33/moments/Squeeze_1:output:0/batch_normalization_33/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_33/batchnorm/RsqrtRsqrt(batch_normalization_33/batchnorm/add:z:0*
T0*
_output_shapes
:м
3batch_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╣
$batch_normalization_33/batchnorm/mulMul*batch_normalization_33/batchnorm/Rsqrt:y:0;batch_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ж
&batch_normalization_33/batchnorm/mul_1Muldense_32/Relu:activations:0(batch_normalization_33/batchnorm/mul:z:0*
T0*'
_output_shapes
:         н
&batch_normalization_33/batchnorm/mul_2Mul/batch_normalization_33/moments/Squeeze:output:0(batch_normalization_33/batchnorm/mul:z:0*
T0*
_output_shapes
:д
/batch_normalization_33/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╡
$batch_normalization_33/batchnorm/subSub7batch_normalization_33/batchnorm/ReadVariableOp:value:0*batch_normalization_33/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╖
&batch_normalization_33/batchnorm/add_1AddV2*batch_normalization_33/batchnorm/mul_1:z:0(batch_normalization_33/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ]
dropout_33/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?Ю
dropout_33/dropout/MulMul*batch_normalization_33/batchnorm/add_1:z:0!dropout_33/dropout/Const:output:0*
T0*'
_output_shapes
:         r
dropout_33/dropout/ShapeShape*batch_normalization_33/batchnorm/add_1:z:0*
T0*
_output_shapes
:в
/dropout_33/dropout/random_uniform/RandomUniformRandomUniform!dropout_33/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0f
!dropout_33/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╟
dropout_33/dropout/GreaterEqualGreaterEqual8dropout_33/dropout/random_uniform/RandomUniform:output:0*dropout_33/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
dropout_33/dropout/CastCast#dropout_33/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         К
dropout_33/dropout/Mul_1Muldropout_33/dropout/Mul:z:0dropout_33/dropout/Cast:y:0*
T0*'
_output_shapes
:         ]
dropout_35/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?Ю
dropout_35/dropout/MulMul*batch_normalization_35/batchnorm/add_1:z:0!dropout_35/dropout/Const:output:0*
T0*'
_output_shapes
:         r
dropout_35/dropout/ShapeShape*batch_normalization_35/batchnorm/add_1:z:0*
T0*
_output_shapes
:в
/dropout_35/dropout/random_uniform/RandomUniformRandomUniform!dropout_35/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0f
!dropout_35/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╟
dropout_35/dropout/GreaterEqualGreaterEqual8dropout_35/dropout/random_uniform/RandomUniform:output:0*dropout_35/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
dropout_35/dropout/CastCast#dropout_35/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         К
dropout_35/dropout/Mul_1Muldropout_35/dropout/Mul:z:0dropout_35/dropout/Cast:y:0*
T0*'
_output_shapes
:         [
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╗
concatenate_8/concatConcatV2dropout_33/dropout/Mul_1:z:0dropout_35/dropout/Mul_1:z:0"concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:         Ж
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Т
dense_35/MatMulMatMulconcatenate_8/concat:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_35/SigmoidSigmoiddense_35/BiasAdd:output:0*
T0*'
_output_shapes
:         c
IdentityIdentitydense_35/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp'^batch_normalization_32/AssignMovingAvg6^batch_normalization_32/AssignMovingAvg/ReadVariableOp)^batch_normalization_32/AssignMovingAvg_18^batch_normalization_32/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_32/batchnorm/ReadVariableOp4^batch_normalization_32/batchnorm/mul/ReadVariableOp'^batch_normalization_33/AssignMovingAvg6^batch_normalization_33/AssignMovingAvg/ReadVariableOp)^batch_normalization_33/AssignMovingAvg_18^batch_normalization_33/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_33/batchnorm/ReadVariableOp4^batch_normalization_33/batchnorm/mul/ReadVariableOp'^batch_normalization_34/AssignMovingAvg6^batch_normalization_34/AssignMovingAvg/ReadVariableOp)^batch_normalization_34/AssignMovingAvg_18^batch_normalization_34/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_34/batchnorm/ReadVariableOp4^batch_normalization_34/batchnorm/mul/ReadVariableOp'^batch_normalization_35/AssignMovingAvg6^batch_normalization_35/AssignMovingAvg/ReadVariableOp)^batch_normalization_35/AssignMovingAvg_18^batch_normalization_35/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_35/batchnorm/ReadVariableOp4^batch_normalization_35/batchnorm/mul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp^embedding_8/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 2P
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
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2<
embedding_8/embedding_lookupembedding_8/embedding_lookup:Q M
'
_output_shapes
:         	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
Ї	
e
F__inference_dropout_33_layer_call_and_return_conditional_losses_920596

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_34_layer_call_and_return_conditional_losses_921793

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╟М
╛+
"__inference__traced_restore_922522
file_prefix9
'assignvariableop_embedding_8_embeddings:=
/assignvariableop_1_batch_normalization_32_gamma:<
.assignvariableop_2_batch_normalization_32_beta:C
5assignvariableop_3_batch_normalization_32_moving_mean:G
9assignvariableop_4_batch_normalization_32_moving_variance:4
"assignvariableop_5_dense_33_kernel:
.
 assignvariableop_6_dense_33_bias:=
/assignvariableop_7_batch_normalization_34_gamma:<
.assignvariableop_8_batch_normalization_34_beta:C
5assignvariableop_9_batch_normalization_34_moving_mean:H
:assignvariableop_10_batch_normalization_34_moving_variance:5
#assignvariableop_11_dense_32_kernel:/
!assignvariableop_12_dense_32_bias:5
#assignvariableop_13_dense_34_kernel:/
!assignvariableop_14_dense_34_bias:>
0assignvariableop_15_batch_normalization_33_gamma:=
/assignvariableop_16_batch_normalization_33_beta:D
6assignvariableop_17_batch_normalization_33_moving_mean:H
:assignvariableop_18_batch_normalization_33_moving_variance:>
0assignvariableop_19_batch_normalization_35_gamma:=
/assignvariableop_20_batch_normalization_35_beta:D
6assignvariableop_21_batch_normalization_35_moving_mean:H
:assignvariableop_22_batch_normalization_35_moving_variance:5
#assignvariableop_23_dense_35_kernel:/
!assignvariableop_24_dense_35_bias:'
assignvariableop_25_adam_iter:	 )
assignvariableop_26_adam_beta_1: )
assignvariableop_27_adam_beta_2: (
assignvariableop_28_adam_decay: 0
&assignvariableop_29_adam_learning_rate: #
assignvariableop_30_total: #
assignvariableop_31_count: C
1assignvariableop_32_adam_embedding_8_embeddings_m:E
7assignvariableop_33_adam_batch_normalization_32_gamma_m:D
6assignvariableop_34_adam_batch_normalization_32_beta_m:<
*assignvariableop_35_adam_dense_33_kernel_m:
6
(assignvariableop_36_adam_dense_33_bias_m:E
7assignvariableop_37_adam_batch_normalization_34_gamma_m:D
6assignvariableop_38_adam_batch_normalization_34_beta_m:<
*assignvariableop_39_adam_dense_32_kernel_m:6
(assignvariableop_40_adam_dense_32_bias_m:<
*assignvariableop_41_adam_dense_34_kernel_m:6
(assignvariableop_42_adam_dense_34_bias_m:E
7assignvariableop_43_adam_batch_normalization_33_gamma_m:D
6assignvariableop_44_adam_batch_normalization_33_beta_m:E
7assignvariableop_45_adam_batch_normalization_35_gamma_m:D
6assignvariableop_46_adam_batch_normalization_35_beta_m:<
*assignvariableop_47_adam_dense_35_kernel_m:6
(assignvariableop_48_adam_dense_35_bias_m:C
1assignvariableop_49_adam_embedding_8_embeddings_v:E
7assignvariableop_50_adam_batch_normalization_32_gamma_v:D
6assignvariableop_51_adam_batch_normalization_32_beta_v:<
*assignvariableop_52_adam_dense_33_kernel_v:
6
(assignvariableop_53_adam_dense_33_bias_v:E
7assignvariableop_54_adam_batch_normalization_34_gamma_v:D
6assignvariableop_55_adam_batch_normalization_34_beta_v:<
*assignvariableop_56_adam_dense_32_kernel_v:6
(assignvariableop_57_adam_dense_32_bias_v:<
*assignvariableop_58_adam_dense_34_kernel_v:6
(assignvariableop_59_adam_dense_34_bias_v:E
7assignvariableop_60_adam_batch_normalization_33_gamma_v:D
6assignvariableop_61_adam_batch_normalization_33_beta_v:E
7assignvariableop_62_adam_batch_normalization_35_gamma_v:D
6assignvariableop_63_adam_batch_normalization_35_beta_v:<
*assignvariableop_64_adam_dense_35_kernel_v:6
(assignvariableop_65_adam_dense_35_bias_v:
identity_67ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Ж%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*м$
valueв$BЯ$CB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH∙
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ё
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*в
_output_shapesП
М:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOpAssignVariableOp'assignvariableop_embedding_8_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_32_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_32_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_32_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_32_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_33_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_33_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_34_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_34_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_9AssignVariableOp5assignvariableop_9_batch_normalization_34_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_10AssignVariableOp:assignvariableop_10_batch_normalization_34_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_32_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_32_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_34_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_34_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_33_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_33_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_17AssignVariableOp6assignvariableop_17_batch_normalization_33_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_18AssignVariableOp:assignvariableop_18_batch_normalization_33_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_19AssignVariableOp0assignvariableop_19_batch_normalization_35_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_35_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_21AssignVariableOp6assignvariableop_21_batch_normalization_35_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_22AssignVariableOp:assignvariableop_22_batch_normalization_35_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_23AssignVariableOp#assignvariableop_23_dense_35_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_24AssignVariableOp!assignvariableop_24_dense_35_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_iterIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_decayIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_learning_rateIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adam_embedding_8_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_batch_normalization_32_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_32_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_33_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_33_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_34_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_34_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_32_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_32_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_34_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_34_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_33_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_33_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_35_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_35_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_35_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_35_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_49AssignVariableOp1assignvariableop_49_adam_embedding_8_embeddings_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_batch_normalization_32_gamma_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_batch_normalization_32_beta_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_dense_33_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_dense_33_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_54AssignVariableOp7assignvariableop_54_adam_batch_normalization_34_gamma_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_34_beta_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_dense_32_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_dense_32_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_34_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_34_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_33_gamma_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_33_beta_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_62AssignVariableOp7assignvariableop_62_adam_batch_normalization_35_gamma_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_batch_normalization_35_beta_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_dense_35_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_dense_35_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 √
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_67IdentityIdentity_66:output:0^NoOp_1*
T0*
_output_shapes
: ш
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_67Identity_67:output:0*Ы
_input_shapesЙ
Ж: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
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
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
аJ
ц
C__inference_model_8_layer_call_and_return_conditional_losses_920991
input_17
input_18$
embedding_8_920925:!
dense_33_920928:

dense_33_920930:+
batch_normalization_32_920933:+
batch_normalization_32_920935:+
batch_normalization_32_920937:+
batch_normalization_32_920939:+
batch_normalization_34_920942:+
batch_normalization_34_920944:+
batch_normalization_34_920946:+
batch_normalization_34_920948:!
dense_34_920954:
dense_34_920956:!
dense_32_920959:
dense_32_920961:+
batch_normalization_35_920964:+
batch_normalization_35_920966:+
batch_normalization_35_920968:+
batch_normalization_35_920970:+
batch_normalization_33_920973:+
batch_normalization_33_920975:+
batch_normalization_33_920977:+
batch_normalization_33_920979:!
dense_35_920985:
dense_35_920987:
identityИв.batch_normalization_32/StatefulPartitionedCallв.batch_normalization_33/StatefulPartitionedCallв.batch_normalization_34/StatefulPartitionedCallв.batch_normalization_35/StatefulPartitionedCallв dense_32/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв dense_34/StatefulPartitionedCallв dense_35/StatefulPartitionedCallв#embedding_8/StatefulPartitionedCallь
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_17embedding_8_920925*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_920336Є
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinput_18dense_33_920928dense_33_920930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_920351Ф
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall,embedding_8/StatefulPartitionedCall:output:0batch_normalization_32_920933batch_normalization_32_920935batch_normalization_32_920937batch_normalization_32_920939*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_920000Н
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0batch_normalization_34_920942batch_normalization_34_920944batch_normalization_34_920946batch_normalization_34_920948*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_920082ё
dropout_32/PartitionedCallPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_920380э
dropout_34/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_920387∙
*global_average_pooling1d_8/PartitionedCallPartitionedCall#dropout_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_920150Н
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0dense_34_920954dense_34_920956*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_920401Э
 dense_32/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_32_920959dense_32_920961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_920418Н
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0batch_normalization_35_920964batch_normalization_35_920966batch_normalization_35_920968batch_normalization_35_920970*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_920259Н
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0batch_normalization_33_920973batch_normalization_33_920975batch_normalization_33_920977batch_normalization_33_920979*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_920177э
dropout_33/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_920447э
dropout_35/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_920454Е
concatenate_8/PartitionedCallPartitionedCall#dropout_33/PartitionedCall:output:0#dropout_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_concatenate_8_layer_call_and_return_conditional_losses_920463Р
 dense_35/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_35_920985dense_35_920987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_920476x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╝
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
input_17:QM
'
_output_shapes
:         

"
_user_specified_name
input_18
У%
ы
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_921925

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
Ц
)__inference_dense_35_layer_call_fn_922081

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_920476o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
У%
ы
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_921767

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▒
G
+__inference_dropout_32_layer_call_fn_921665

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_920380d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
Ї	
e
F__inference_dropout_35_layer_call_and_return_conditional_losses_922059

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
╥
7__inference_batch_normalization_33_layer_call_fn_921858

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_920177o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_33_layer_call_and_return_conditional_losses_922020

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ї	
e
F__inference_dropout_34_layer_call_and_return_conditional_losses_921805

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_33_layer_call_and_return_conditional_losses_920447

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
Ц
)__inference_dense_32_layer_call_fn_921814

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_920418o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
щ
d
F__inference_dropout_32_layer_call_and_return_conditional_losses_921675

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         	_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         	"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
Ы

ї
D__inference_dense_32_layer_call_and_return_conditional_losses_921825

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ф

e
F__inference_dropout_32_layer_call_and_return_conditional_losses_921687

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         	C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         	*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         	s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         	m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         	]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
Ф

e
F__inference_dropout_32_layer_call_and_return_conditional_losses_920662

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         	C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         	*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         	s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         	m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         	]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
╧
▒
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_921971

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ї	
e
F__inference_dropout_35_layer_call_and_return_conditional_losses_920573

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы

ї
D__inference_dense_32_layer_call_and_return_conditional_losses_920418

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
є
d
+__inference_dropout_34_layer_call_fn_921788

inputs
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_920639o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
г	
д
G__inference_embedding_8_layer_call_and_return_conditional_losses_920336

inputs)
embedding_lookup_920330:
identityИвembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         	╗
embedding_lookupResourceGatherembedding_lookup_920330Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/920330*+
_output_shapes
:         	*
dtype0в
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/920330*+
_output_shapes
:         	Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         	w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:         	Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         	: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
Ы

ї
D__inference_dense_33_layer_call_and_return_conditional_losses_921660

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Р
▒
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_920000

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Й
В
(__inference_model_8_layer_call_fn_920536
input_17
input_18
unknown:
	unknown_0:

	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identityИвStatefulPartitionedCallЮ
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
unknown_23*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_920483o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
input_17:QM
'
_output_shapes
:         

"
_user_specified_name
input_18
ДШ
А
C__inference_model_8_layer_call_and_return_conditional_losses_921290
inputs_0
inputs_15
#embedding_8_embedding_lookup_921184:9
'dense_33_matmul_readvariableop_resource:
6
(dense_33_biasadd_readvariableop_resource:F
8batch_normalization_32_batchnorm_readvariableop_resource:J
<batch_normalization_32_batchnorm_mul_readvariableop_resource:H
:batch_normalization_32_batchnorm_readvariableop_1_resource:H
:batch_normalization_32_batchnorm_readvariableop_2_resource:F
8batch_normalization_34_batchnorm_readvariableop_resource:J
<batch_normalization_34_batchnorm_mul_readvariableop_resource:H
:batch_normalization_34_batchnorm_readvariableop_1_resource:H
:batch_normalization_34_batchnorm_readvariableop_2_resource:9
'dense_34_matmul_readvariableop_resource:6
(dense_34_biasadd_readvariableop_resource:9
'dense_32_matmul_readvariableop_resource:6
(dense_32_biasadd_readvariableop_resource:F
8batch_normalization_35_batchnorm_readvariableop_resource:J
<batch_normalization_35_batchnorm_mul_readvariableop_resource:H
:batch_normalization_35_batchnorm_readvariableop_1_resource:H
:batch_normalization_35_batchnorm_readvariableop_2_resource:F
8batch_normalization_33_batchnorm_readvariableop_resource:J
<batch_normalization_33_batchnorm_mul_readvariableop_resource:H
:batch_normalization_33_batchnorm_readvariableop_1_resource:H
:batch_normalization_33_batchnorm_readvariableop_2_resource:9
'dense_35_matmul_readvariableop_resource:6
(dense_35_biasadd_readvariableop_resource:
identityИв/batch_normalization_32/batchnorm/ReadVariableOpв1batch_normalization_32/batchnorm/ReadVariableOp_1в1batch_normalization_32/batchnorm/ReadVariableOp_2в3batch_normalization_32/batchnorm/mul/ReadVariableOpв/batch_normalization_33/batchnorm/ReadVariableOpв1batch_normalization_33/batchnorm/ReadVariableOp_1в1batch_normalization_33/batchnorm/ReadVariableOp_2в3batch_normalization_33/batchnorm/mul/ReadVariableOpв/batch_normalization_34/batchnorm/ReadVariableOpв1batch_normalization_34/batchnorm/ReadVariableOp_1в1batch_normalization_34/batchnorm/ReadVariableOp_2в3batch_normalization_34/batchnorm/mul/ReadVariableOpв/batch_normalization_35/batchnorm/ReadVariableOpв1batch_normalization_35/batchnorm/ReadVariableOp_1в1batch_normalization_35/batchnorm/ReadVariableOp_2в3batch_normalization_35/batchnorm/mul/ReadVariableOpвdense_32/BiasAdd/ReadVariableOpвdense_32/MatMul/ReadVariableOpвdense_33/BiasAdd/ReadVariableOpвdense_33/MatMul/ReadVariableOpвdense_34/BiasAdd/ReadVariableOpвdense_34/MatMul/ReadVariableOpвdense_35/BiasAdd/ReadVariableOpвdense_35/MatMul/ReadVariableOpвembedding_8/embedding_lookupc
embedding_8/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:         	ы
embedding_8/embedding_lookupResourceGather#embedding_8_embedding_lookup_921184embedding_8/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_8/embedding_lookup/921184*+
_output_shapes
:         	*
dtype0╞
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_8/embedding_lookup/921184*+
_output_shapes
:         	Щ
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         	Ж
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0}
dense_33/MatMulMatMulinputs_1&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:         д
/batch_normalization_32/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╝
$batch_normalization_32/batchnorm/addAddV27batch_normalization_32/batchnorm/ReadVariableOp:value:0/batch_normalization_32/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_32/batchnorm/RsqrtRsqrt(batch_normalization_32/batchnorm/add:z:0*
T0*
_output_shapes
:м
3batch_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╣
$batch_normalization_32/batchnorm/mulMul*batch_normalization_32/batchnorm/Rsqrt:y:0;batch_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:┐
&batch_normalization_32/batchnorm/mul_1Mul0embedding_8/embedding_lookup/Identity_1:output:0(batch_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:         	и
1batch_normalization_32/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_32_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╖
&batch_normalization_32/batchnorm/mul_2Mul9batch_normalization_32/batchnorm/ReadVariableOp_1:value:0(batch_normalization_32/batchnorm/mul:z:0*
T0*
_output_shapes
:и
1batch_normalization_32/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_32_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╖
$batch_normalization_32/batchnorm/subSub9batch_normalization_32/batchnorm/ReadVariableOp_2:value:0*batch_normalization_32/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╗
&batch_normalization_32/batchnorm/add_1AddV2*batch_normalization_32/batchnorm/mul_1:z:0(batch_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:         	д
/batch_normalization_34/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_34_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_34/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╝
$batch_normalization_34/batchnorm/addAddV27batch_normalization_34/batchnorm/ReadVariableOp:value:0/batch_normalization_34/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_34/batchnorm/RsqrtRsqrt(batch_normalization_34/batchnorm/add:z:0*
T0*
_output_shapes
:м
3batch_normalization_34/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_34_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╣
$batch_normalization_34/batchnorm/mulMul*batch_normalization_34/batchnorm/Rsqrt:y:0;batch_normalization_34/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ж
&batch_normalization_34/batchnorm/mul_1Muldense_33/Relu:activations:0(batch_normalization_34/batchnorm/mul:z:0*
T0*'
_output_shapes
:         и
1batch_normalization_34/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_34_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╖
&batch_normalization_34/batchnorm/mul_2Mul9batch_normalization_34/batchnorm/ReadVariableOp_1:value:0(batch_normalization_34/batchnorm/mul:z:0*
T0*
_output_shapes
:и
1batch_normalization_34/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_34_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╖
$batch_normalization_34/batchnorm/subSub9batch_normalization_34/batchnorm/ReadVariableOp_2:value:0*batch_normalization_34/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╖
&batch_normalization_34/batchnorm/add_1AddV2*batch_normalization_34/batchnorm/mul_1:z:0(batch_normalization_34/batchnorm/sub:z:0*
T0*'
_output_shapes
:         Б
dropout_32/IdentityIdentity*batch_normalization_32/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         	}
dropout_34/IdentityIdentity*batch_normalization_34/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         s
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :│
global_average_pooling1d_8/MeanMeandropout_32/Identity:output:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Ж
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:*
dtype0С
dense_34/MatMulMatMuldropout_34/Identity:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:         Ж
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Э
dense_32/MatMulMatMul(global_average_pooling1d_8/Mean:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:         д
/batch_normalization_35/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_35_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_35/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╝
$batch_normalization_35/batchnorm/addAddV27batch_normalization_35/batchnorm/ReadVariableOp:value:0/batch_normalization_35/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_35/batchnorm/RsqrtRsqrt(batch_normalization_35/batchnorm/add:z:0*
T0*
_output_shapes
:м
3batch_normalization_35/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_35_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╣
$batch_normalization_35/batchnorm/mulMul*batch_normalization_35/batchnorm/Rsqrt:y:0;batch_normalization_35/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ж
&batch_normalization_35/batchnorm/mul_1Muldense_34/Relu:activations:0(batch_normalization_35/batchnorm/mul:z:0*
T0*'
_output_shapes
:         и
1batch_normalization_35/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_35_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╖
&batch_normalization_35/batchnorm/mul_2Mul9batch_normalization_35/batchnorm/ReadVariableOp_1:value:0(batch_normalization_35/batchnorm/mul:z:0*
T0*
_output_shapes
:и
1batch_normalization_35/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_35_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╖
$batch_normalization_35/batchnorm/subSub9batch_normalization_35/batchnorm/ReadVariableOp_2:value:0*batch_normalization_35/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╖
&batch_normalization_35/batchnorm/add_1AddV2*batch_normalization_35/batchnorm/mul_1:z:0(batch_normalization_35/batchnorm/sub:z:0*
T0*'
_output_shapes
:         д
/batch_normalization_33/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╝
$batch_normalization_33/batchnorm/addAddV27batch_normalization_33/batchnorm/ReadVariableOp:value:0/batch_normalization_33/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_33/batchnorm/RsqrtRsqrt(batch_normalization_33/batchnorm/add:z:0*
T0*
_output_shapes
:м
3batch_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╣
$batch_normalization_33/batchnorm/mulMul*batch_normalization_33/batchnorm/Rsqrt:y:0;batch_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ж
&batch_normalization_33/batchnorm/mul_1Muldense_32/Relu:activations:0(batch_normalization_33/batchnorm/mul:z:0*
T0*'
_output_shapes
:         и
1batch_normalization_33/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_33_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╖
&batch_normalization_33/batchnorm/mul_2Mul9batch_normalization_33/batchnorm/ReadVariableOp_1:value:0(batch_normalization_33/batchnorm/mul:z:0*
T0*
_output_shapes
:и
1batch_normalization_33/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_33_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╖
$batch_normalization_33/batchnorm/subSub9batch_normalization_33/batchnorm/ReadVariableOp_2:value:0*batch_normalization_33/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╖
&batch_normalization_33/batchnorm/add_1AddV2*batch_normalization_33/batchnorm/mul_1:z:0(batch_normalization_33/batchnorm/sub:z:0*
T0*'
_output_shapes
:         }
dropout_33/IdentityIdentity*batch_normalization_33/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         }
dropout_35/IdentityIdentity*batch_normalization_35/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         [
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╗
concatenate_8/concatConcatV2dropout_33/Identity:output:0dropout_35/Identity:output:0"concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:         Ж
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Т
dense_35/MatMulMatMulconcatenate_8/concat:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_35/SigmoidSigmoiddense_35/BiasAdd:output:0*
T0*'
_output_shapes
:         c
IdentityIdentitydense_35/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         ▒	
NoOpNoOp0^batch_normalization_32/batchnorm/ReadVariableOp2^batch_normalization_32/batchnorm/ReadVariableOp_12^batch_normalization_32/batchnorm/ReadVariableOp_24^batch_normalization_32/batchnorm/mul/ReadVariableOp0^batch_normalization_33/batchnorm/ReadVariableOp2^batch_normalization_33/batchnorm/ReadVariableOp_12^batch_normalization_33/batchnorm/ReadVariableOp_24^batch_normalization_33/batchnorm/mul/ReadVariableOp0^batch_normalization_34/batchnorm/ReadVariableOp2^batch_normalization_34/batchnorm/ReadVariableOp_12^batch_normalization_34/batchnorm/ReadVariableOp_24^batch_normalization_34/batchnorm/mul/ReadVariableOp0^batch_normalization_35/batchnorm/ReadVariableOp2^batch_normalization_35/batchnorm/ReadVariableOp_12^batch_normalization_35/batchnorm/ReadVariableOp_24^batch_normalization_35/batchnorm/mul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp^embedding_8/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 2b
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
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2<
embedding_8/embedding_lookupembedding_8/embedding_lookup:Q M
'
_output_shapes
:         	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
Ї	
e
F__inference_dropout_34_layer_call_and_return_conditional_losses_920639

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Р
▒
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_921606

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
б
G
+__inference_dropout_34_layer_call_fn_921783

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_920387`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ї	
e
F__inference_dropout_33_layer_call_and_return_conditional_losses_922032

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
О
r
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_920150

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
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
є
d
+__inference_dropout_35_layer_call_fn_922042

inputs
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_920573o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╧
▒
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_920259

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ШJ
ф
C__inference_model_8_layer_call_and_return_conditional_losses_920483

inputs
inputs_1$
embedding_8_920337:!
dense_33_920352:

dense_33_920354:+
batch_normalization_32_920357:+
batch_normalization_32_920359:+
batch_normalization_32_920361:+
batch_normalization_32_920363:+
batch_normalization_34_920366:+
batch_normalization_34_920368:+
batch_normalization_34_920370:+
batch_normalization_34_920372:!
dense_34_920402:
dense_34_920404:!
dense_32_920419:
dense_32_920421:+
batch_normalization_35_920424:+
batch_normalization_35_920426:+
batch_normalization_35_920428:+
batch_normalization_35_920430:+
batch_normalization_33_920433:+
batch_normalization_33_920435:+
batch_normalization_33_920437:+
batch_normalization_33_920439:!
dense_35_920477:
dense_35_920479:
identityИв.batch_normalization_32/StatefulPartitionedCallв.batch_normalization_33/StatefulPartitionedCallв.batch_normalization_34/StatefulPartitionedCallв.batch_normalization_35/StatefulPartitionedCallв dense_32/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв dense_34/StatefulPartitionedCallв dense_35/StatefulPartitionedCallв#embedding_8/StatefulPartitionedCallъ
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_8_920337*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_920336Є
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_33_920352dense_33_920354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_920351Ф
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall,embedding_8/StatefulPartitionedCall:output:0batch_normalization_32_920357batch_normalization_32_920359batch_normalization_32_920361batch_normalization_32_920363*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_920000Н
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0batch_normalization_34_920366batch_normalization_34_920368batch_normalization_34_920370batch_normalization_34_920372*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_920082ё
dropout_32/PartitionedCallPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_920380э
dropout_34/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_920387∙
*global_average_pooling1d_8/PartitionedCallPartitionedCall#dropout_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_920150Н
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0dense_34_920402dense_34_920404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_920401Э
 dense_32/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_32_920419dense_32_920421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_920418Н
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0batch_normalization_35_920424batch_normalization_35_920426batch_normalization_35_920428batch_normalization_35_920430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_920259Н
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0batch_normalization_33_920433batch_normalization_33_920435batch_normalization_33_920437batch_normalization_33_920439*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_920177э
dropout_33/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_920447э
dropout_35/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_920454Е
concatenate_8/PartitionedCallPartitionedCall#dropout_33/PartitionedCall:output:0#dropout_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_concatenate_8_layer_call_and_return_conditional_losses_920463Р
 dense_35/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_35_920477dense_35_920479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_920476x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╝
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         

 
_user_specified_nameinputs
О
r
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_921778

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
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
┘
d
F__inference_dropout_34_layer_call_and_return_conditional_losses_920387

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
╥
7__inference_batch_normalization_34_layer_call_fn_921700

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_920082o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
╥
7__inference_batch_normalization_33_layer_call_fn_921871

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_920224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
щ
d
F__inference_dropout_32_layer_call_and_return_conditional_losses_920380

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         	_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         	"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
У%
ы
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_920306

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Б
В
(__inference_model_8_layer_call_fn_921179
inputs_0
inputs_1
unknown:
	unknown_0:

	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identityИвStatefulPartitionedCallЦ
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
unknown_23*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_920812o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
■%
ы
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_921640

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       в
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
У%
ы
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_920224

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы

ї
D__inference_dense_34_layer_call_and_return_conditional_losses_921845

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
╥
7__inference_batch_normalization_35_layer_call_fn_921938

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_920259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╧
▒
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_920177

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╩л
ю
!__inference__wrapped_model_919976
input_17
input_18=
+model_8_embedding_8_embedding_lookup_919870:A
/model_8_dense_33_matmul_readvariableop_resource:
>
0model_8_dense_33_biasadd_readvariableop_resource:N
@model_8_batch_normalization_32_batchnorm_readvariableop_resource:R
Dmodel_8_batch_normalization_32_batchnorm_mul_readvariableop_resource:P
Bmodel_8_batch_normalization_32_batchnorm_readvariableop_1_resource:P
Bmodel_8_batch_normalization_32_batchnorm_readvariableop_2_resource:N
@model_8_batch_normalization_34_batchnorm_readvariableop_resource:R
Dmodel_8_batch_normalization_34_batchnorm_mul_readvariableop_resource:P
Bmodel_8_batch_normalization_34_batchnorm_readvariableop_1_resource:P
Bmodel_8_batch_normalization_34_batchnorm_readvariableop_2_resource:A
/model_8_dense_34_matmul_readvariableop_resource:>
0model_8_dense_34_biasadd_readvariableop_resource:A
/model_8_dense_32_matmul_readvariableop_resource:>
0model_8_dense_32_biasadd_readvariableop_resource:N
@model_8_batch_normalization_35_batchnorm_readvariableop_resource:R
Dmodel_8_batch_normalization_35_batchnorm_mul_readvariableop_resource:P
Bmodel_8_batch_normalization_35_batchnorm_readvariableop_1_resource:P
Bmodel_8_batch_normalization_35_batchnorm_readvariableop_2_resource:N
@model_8_batch_normalization_33_batchnorm_readvariableop_resource:R
Dmodel_8_batch_normalization_33_batchnorm_mul_readvariableop_resource:P
Bmodel_8_batch_normalization_33_batchnorm_readvariableop_1_resource:P
Bmodel_8_batch_normalization_33_batchnorm_readvariableop_2_resource:A
/model_8_dense_35_matmul_readvariableop_resource:>
0model_8_dense_35_biasadd_readvariableop_resource:
identityИв7model_8/batch_normalization_32/batchnorm/ReadVariableOpв9model_8/batch_normalization_32/batchnorm/ReadVariableOp_1в9model_8/batch_normalization_32/batchnorm/ReadVariableOp_2в;model_8/batch_normalization_32/batchnorm/mul/ReadVariableOpв7model_8/batch_normalization_33/batchnorm/ReadVariableOpв9model_8/batch_normalization_33/batchnorm/ReadVariableOp_1в9model_8/batch_normalization_33/batchnorm/ReadVariableOp_2в;model_8/batch_normalization_33/batchnorm/mul/ReadVariableOpв7model_8/batch_normalization_34/batchnorm/ReadVariableOpв9model_8/batch_normalization_34/batchnorm/ReadVariableOp_1в9model_8/batch_normalization_34/batchnorm/ReadVariableOp_2в;model_8/batch_normalization_34/batchnorm/mul/ReadVariableOpв7model_8/batch_normalization_35/batchnorm/ReadVariableOpв9model_8/batch_normalization_35/batchnorm/ReadVariableOp_1в9model_8/batch_normalization_35/batchnorm/ReadVariableOp_2в;model_8/batch_normalization_35/batchnorm/mul/ReadVariableOpв'model_8/dense_32/BiasAdd/ReadVariableOpв&model_8/dense_32/MatMul/ReadVariableOpв'model_8/dense_33/BiasAdd/ReadVariableOpв&model_8/dense_33/MatMul/ReadVariableOpв'model_8/dense_34/BiasAdd/ReadVariableOpв&model_8/dense_34/MatMul/ReadVariableOpв'model_8/dense_35/BiasAdd/ReadVariableOpв&model_8/dense_35/MatMul/ReadVariableOpв$model_8/embedding_8/embedding_lookupk
model_8/embedding_8/CastCastinput_17*

DstT0*

SrcT0*'
_output_shapes
:         	Л
$model_8/embedding_8/embedding_lookupResourceGather+model_8_embedding_8_embedding_lookup_919870model_8/embedding_8/Cast:y:0*
Tindices0*>
_class4
20loc:@model_8/embedding_8/embedding_lookup/919870*+
_output_shapes
:         	*
dtype0▐
-model_8/embedding_8/embedding_lookup/IdentityIdentity-model_8/embedding_8/embedding_lookup:output:0*
T0*>
_class4
20loc:@model_8/embedding_8/embedding_lookup/919870*+
_output_shapes
:         	й
/model_8/embedding_8/embedding_lookup/Identity_1Identity6model_8/embedding_8/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         	Ц
&model_8/dense_33/MatMul/ReadVariableOpReadVariableOp/model_8_dense_33_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Н
model_8/dense_33/MatMulMatMulinput_18.model_8/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
'model_8/dense_33/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0й
model_8/dense_33/BiasAddBiasAdd!model_8/dense_33/MatMul:product:0/model_8/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
model_8/dense_33/ReluRelu!model_8/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:         ┤
7model_8/batch_normalization_32/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0s
.model_8/batch_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╘
,model_8/batch_normalization_32/batchnorm/addAddV2?model_8/batch_normalization_32/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_32/batchnorm/add/y:output:0*
T0*
_output_shapes
:О
.model_8/batch_normalization_32/batchnorm/RsqrtRsqrt0model_8/batch_normalization_32/batchnorm/add:z:0*
T0*
_output_shapes
:╝
;model_8/batch_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╤
,model_8/batch_normalization_32/batchnorm/mulMul2model_8/batch_normalization_32/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:╫
.model_8/batch_normalization_32/batchnorm/mul_1Mul8model_8/embedding_8/embedding_lookup/Identity_1:output:00model_8/batch_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:         	╕
9model_8/batch_normalization_32/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_32_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╧
.model_8/batch_normalization_32/batchnorm/mul_2MulAmodel_8/batch_normalization_32/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_32/batchnorm/mul:z:0*
T0*
_output_shapes
:╕
9model_8/batch_normalization_32/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_32_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╧
,model_8/batch_normalization_32/batchnorm/subSubAmodel_8/batch_normalization_32/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_32/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╙
.model_8/batch_normalization_32/batchnorm/add_1AddV22model_8/batch_normalization_32/batchnorm/mul_1:z:00model_8/batch_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:         	┤
7model_8/batch_normalization_34/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_34_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0s
.model_8/batch_normalization_34/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╘
,model_8/batch_normalization_34/batchnorm/addAddV2?model_8/batch_normalization_34/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_34/batchnorm/add/y:output:0*
T0*
_output_shapes
:О
.model_8/batch_normalization_34/batchnorm/RsqrtRsqrt0model_8/batch_normalization_34/batchnorm/add:z:0*
T0*
_output_shapes
:╝
;model_8/batch_normalization_34/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_34_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╤
,model_8/batch_normalization_34/batchnorm/mulMul2model_8/batch_normalization_34/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_34/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:╛
.model_8/batch_normalization_34/batchnorm/mul_1Mul#model_8/dense_33/Relu:activations:00model_8/batch_normalization_34/batchnorm/mul:z:0*
T0*'
_output_shapes
:         ╕
9model_8/batch_normalization_34/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_34_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╧
.model_8/batch_normalization_34/batchnorm/mul_2MulAmodel_8/batch_normalization_34/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_34/batchnorm/mul:z:0*
T0*
_output_shapes
:╕
9model_8/batch_normalization_34/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_34_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╧
,model_8/batch_normalization_34/batchnorm/subSubAmodel_8/batch_normalization_34/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_34/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╧
.model_8/batch_normalization_34/batchnorm/add_1AddV22model_8/batch_normalization_34/batchnorm/mul_1:z:00model_8/batch_normalization_34/batchnorm/sub:z:0*
T0*'
_output_shapes
:         С
model_8/dropout_32/IdentityIdentity2model_8/batch_normalization_32/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         	Н
model_8/dropout_34/IdentityIdentity2model_8/batch_normalization_34/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         {
9model_8/global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :╦
'model_8/global_average_pooling1d_8/MeanMean$model_8/dropout_32/Identity:output:0Bmodel_8/global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Ц
&model_8/dense_34/MatMul/ReadVariableOpReadVariableOp/model_8_dense_34_matmul_readvariableop_resource*
_output_shapes

:*
dtype0й
model_8/dense_34/MatMulMatMul$model_8/dropout_34/Identity:output:0.model_8/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
'model_8/dense_34/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0й
model_8/dense_34/BiasAddBiasAdd!model_8/dense_34/MatMul:product:0/model_8/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
model_8/dense_34/ReluRelu!model_8/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:         Ц
&model_8/dense_32/MatMul/ReadVariableOpReadVariableOp/model_8_dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╡
model_8/dense_32/MatMulMatMul0model_8/global_average_pooling1d_8/Mean:output:0.model_8/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
'model_8/dense_32/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0й
model_8/dense_32/BiasAddBiasAdd!model_8/dense_32/MatMul:product:0/model_8/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
model_8/dense_32/ReluRelu!model_8/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:         ┤
7model_8/batch_normalization_35/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_35_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0s
.model_8/batch_normalization_35/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╘
,model_8/batch_normalization_35/batchnorm/addAddV2?model_8/batch_normalization_35/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_35/batchnorm/add/y:output:0*
T0*
_output_shapes
:О
.model_8/batch_normalization_35/batchnorm/RsqrtRsqrt0model_8/batch_normalization_35/batchnorm/add:z:0*
T0*
_output_shapes
:╝
;model_8/batch_normalization_35/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_35_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╤
,model_8/batch_normalization_35/batchnorm/mulMul2model_8/batch_normalization_35/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_35/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:╛
.model_8/batch_normalization_35/batchnorm/mul_1Mul#model_8/dense_34/Relu:activations:00model_8/batch_normalization_35/batchnorm/mul:z:0*
T0*'
_output_shapes
:         ╕
9model_8/batch_normalization_35/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_35_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╧
.model_8/batch_normalization_35/batchnorm/mul_2MulAmodel_8/batch_normalization_35/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_35/batchnorm/mul:z:0*
T0*
_output_shapes
:╕
9model_8/batch_normalization_35/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_35_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╧
,model_8/batch_normalization_35/batchnorm/subSubAmodel_8/batch_normalization_35/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_35/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╧
.model_8/batch_normalization_35/batchnorm/add_1AddV22model_8/batch_normalization_35/batchnorm/mul_1:z:00model_8/batch_normalization_35/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ┤
7model_8/batch_normalization_33/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0s
.model_8/batch_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╘
,model_8/batch_normalization_33/batchnorm/addAddV2?model_8/batch_normalization_33/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_33/batchnorm/add/y:output:0*
T0*
_output_shapes
:О
.model_8/batch_normalization_33/batchnorm/RsqrtRsqrt0model_8/batch_normalization_33/batchnorm/add:z:0*
T0*
_output_shapes
:╝
;model_8/batch_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╤
,model_8/batch_normalization_33/batchnorm/mulMul2model_8/batch_normalization_33/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:╛
.model_8/batch_normalization_33/batchnorm/mul_1Mul#model_8/dense_32/Relu:activations:00model_8/batch_normalization_33/batchnorm/mul:z:0*
T0*'
_output_shapes
:         ╕
9model_8/batch_normalization_33/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_33_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╧
.model_8/batch_normalization_33/batchnorm/mul_2MulAmodel_8/batch_normalization_33/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_33/batchnorm/mul:z:0*
T0*
_output_shapes
:╕
9model_8/batch_normalization_33/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_33_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╧
,model_8/batch_normalization_33/batchnorm/subSubAmodel_8/batch_normalization_33/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_33/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╧
.model_8/batch_normalization_33/batchnorm/add_1AddV22model_8/batch_normalization_33/batchnorm/mul_1:z:00model_8/batch_normalization_33/batchnorm/sub:z:0*
T0*'
_output_shapes
:         Н
model_8/dropout_33/IdentityIdentity2model_8/batch_normalization_33/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         Н
model_8/dropout_35/IdentityIdentity2model_8/batch_normalization_35/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         c
!model_8/concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :█
model_8/concatenate_8/concatConcatV2$model_8/dropout_33/Identity:output:0$model_8/dropout_35/Identity:output:0*model_8/concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:         Ц
&model_8/dense_35/MatMul/ReadVariableOpReadVariableOp/model_8_dense_35_matmul_readvariableop_resource*
_output_shapes

:*
dtype0к
model_8/dense_35/MatMulMatMul%model_8/concatenate_8/concat:output:0.model_8/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
'model_8/dense_35/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0й
model_8/dense_35/BiasAddBiasAdd!model_8/dense_35/MatMul:product:0/model_8/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
model_8/dense_35/SigmoidSigmoid!model_8/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:         k
IdentityIdentitymodel_8/dense_35/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         ∙

NoOpNoOp8^model_8/batch_normalization_32/batchnorm/ReadVariableOp:^model_8/batch_normalization_32/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_32/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_32/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_33/batchnorm/ReadVariableOp:^model_8/batch_normalization_33/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_33/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_33/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_34/batchnorm/ReadVariableOp:^model_8/batch_normalization_34/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_34/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_34/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_35/batchnorm/ReadVariableOp:^model_8/batch_normalization_35/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_35/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_35/batchnorm/mul/ReadVariableOp(^model_8/dense_32/BiasAdd/ReadVariableOp'^model_8/dense_32/MatMul/ReadVariableOp(^model_8/dense_33/BiasAdd/ReadVariableOp'^model_8/dense_33/MatMul/ReadVariableOp(^model_8/dense_34/BiasAdd/ReadVariableOp'^model_8/dense_34/MatMul/ReadVariableOp(^model_8/dense_35/BiasAdd/ReadVariableOp'^model_8/dense_35/MatMul/ReadVariableOp%^model_8/embedding_8/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 2r
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
'model_8/dense_32/BiasAdd/ReadVariableOp'model_8/dense_32/BiasAdd/ReadVariableOp2P
&model_8/dense_32/MatMul/ReadVariableOp&model_8/dense_32/MatMul/ReadVariableOp2R
'model_8/dense_33/BiasAdd/ReadVariableOp'model_8/dense_33/BiasAdd/ReadVariableOp2P
&model_8/dense_33/MatMul/ReadVariableOp&model_8/dense_33/MatMul/ReadVariableOp2R
'model_8/dense_34/BiasAdd/ReadVariableOp'model_8/dense_34/BiasAdd/ReadVariableOp2P
&model_8/dense_34/MatMul/ReadVariableOp&model_8/dense_34/MatMul/ReadVariableOp2R
'model_8/dense_35/BiasAdd/ReadVariableOp'model_8/dense_35/BiasAdd/ReadVariableOp2P
&model_8/dense_35/MatMul/ReadVariableOp&model_8/dense_35/MatMul/ReadVariableOp2L
$model_8/embedding_8/embedding_lookup$model_8/embedding_8/embedding_lookup:Q M
'
_output_shapes
:         	
"
_user_specified_name
input_17:QM
'
_output_shapes
:         

"
_user_specified_name
input_18
Ъ

ї
D__inference_dense_35_layer_call_and_return_conditional_losses_922092

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
м
Z
.__inference_concatenate_8_layer_call_fn_922065
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_concatenate_8_layer_call_and_return_conditional_losses_920463`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
▐
╥
7__inference_batch_normalization_32_layer_call_fn_921573

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_920000|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ъ

ї
D__inference_dense_35_layer_call_and_return_conditional_losses_920476

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╧
▒
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_920082

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
Ц
)__inference_dense_34_layer_call_fn_921834

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_920401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╧
▒
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_921891

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы

ї
D__inference_dense_33_layer_call_and_return_conditional_losses_920351

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
╧
▒
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_921733

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Б
В
(__inference_model_8_layer_call_fn_920921
input_17
input_18
unknown:
	unknown_0:

	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identityИвStatefulPartitionedCallЦ
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
unknown_23*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_920812o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
input_17:QM
'
_output_shapes
:         

"
_user_specified_name
input_18
є
d
+__inference_dropout_33_layer_call_fn_922015

inputs
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_920596o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
Ц
)__inference_dense_33_layer_call_fn_921649

inputs
unknown:

	unknown_0:
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_920351o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
▄
╥
7__inference_batch_normalization_32_layer_call_fn_921586

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_920047|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
У%
ы
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_922005

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
u
I__inference_concatenate_8_layer_call_and_return_conditional_losses_922072
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
:         W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
и
╥
7__inference_batch_normalization_34_layer_call_fn_921713

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_920129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╓Е
ч
__inference__traced_save_922314
file_prefix5
1savev2_embedding_8_embeddings_read_readvariableop;
7savev2_batch_normalization_32_gamma_read_readvariableop:
6savev2_batch_normalization_32_beta_read_readvariableopA
=savev2_batch_normalization_32_moving_mean_read_readvariableopE
Asavev2_batch_normalization_32_moving_variance_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop;
7savev2_batch_normalization_34_gamma_read_readvariableop:
6savev2_batch_normalization_34_beta_read_readvariableopA
=savev2_batch_normalization_34_moving_mean_read_readvariableopE
Asavev2_batch_normalization_34_moving_variance_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop;
7savev2_batch_normalization_33_gamma_read_readvariableop:
6savev2_batch_normalization_33_beta_read_readvariableopA
=savev2_batch_normalization_33_moving_mean_read_readvariableopE
Asavev2_batch_normalization_33_moving_variance_read_readvariableop;
7savev2_batch_normalization_35_gamma_read_readvariableop:
6savev2_batch_normalization_35_beta_read_readvariableopA
=savev2_batch_normalization_35_moving_mean_read_readvariableopE
Asavev2_batch_normalization_35_moving_variance_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_embedding_8_embeddings_m_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_m_read_readvariableopB
>savev2_adam_batch_normalization_35_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_35_beta_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop<
8savev2_adam_embedding_8_embeddings_v_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_v_read_readvariableopB
>savev2_adam_batch_normalization_35_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_35_beta_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Г%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*м$
valueв$BЯ$CB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЎ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B х
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_8_embeddings_read_readvariableop7savev2_batch_normalization_32_gamma_read_readvariableop6savev2_batch_normalization_32_beta_read_readvariableop=savev2_batch_normalization_32_moving_mean_read_readvariableopAsavev2_batch_normalization_32_moving_variance_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop7savev2_batch_normalization_34_gamma_read_readvariableop6savev2_batch_normalization_34_beta_read_readvariableop=savev2_batch_normalization_34_moving_mean_read_readvariableopAsavev2_batch_normalization_34_moving_variance_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop7savev2_batch_normalization_33_gamma_read_readvariableop6savev2_batch_normalization_33_beta_read_readvariableop=savev2_batch_normalization_33_moving_mean_read_readvariableopAsavev2_batch_normalization_33_moving_variance_read_readvariableop7savev2_batch_normalization_35_gamma_read_readvariableop6savev2_batch_normalization_35_beta_read_readvariableop=savev2_batch_normalization_35_moving_mean_read_readvariableopAsavev2_batch_normalization_35_moving_variance_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_8_embeddings_m_read_readvariableop>savev2_adam_batch_normalization_32_gamma_m_read_readvariableop=savev2_adam_batch_normalization_32_beta_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop>savev2_adam_batch_normalization_34_gamma_m_read_readvariableop=savev2_adam_batch_normalization_34_beta_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop>savev2_adam_batch_normalization_33_gamma_m_read_readvariableop=savev2_adam_batch_normalization_33_beta_m_read_readvariableop>savev2_adam_batch_normalization_35_gamma_m_read_readvariableop=savev2_adam_batch_normalization_35_beta_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop8savev2_adam_embedding_8_embeddings_v_read_readvariableop>savev2_adam_batch_normalization_32_gamma_v_read_readvariableop=savev2_adam_batch_normalization_32_beta_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableop>savev2_adam_batch_normalization_34_gamma_v_read_readvariableop=savev2_adam_batch_normalization_34_beta_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop>savev2_adam_batch_normalization_33_gamma_v_read_readvariableop=savev2_adam_batch_normalization_33_beta_v_read_readvariableop>savev2_adam_batch_normalization_35_gamma_v_read_readvariableop=savev2_adam_batch_normalization_35_beta_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Q
dtypesG
E2C	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*┼
_input_shapes│
░: ::::::
:::::::::::::::::::: : : : : : : ::::
:::::::::::::::::
:::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :
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
: : 

_output_shapes
: :$! 

_output_shapes

:: "

_output_shapes
:: #

_output_shapes
::$$ 

_output_shapes

:
: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
:: 4

_output_shapes
::$5 

_output_shapes

:
: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
::$9 

_output_shapes

:: :

_output_shapes
::$; 

_output_shapes

:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
::$A 

_output_shapes

:: B

_output_shapes
::C

_output_shapes
: 
б
G
+__inference_dropout_35_layer_call_fn_922037

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_920454`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Г
d
+__inference_dropout_32_layer_call_fn_921670

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_920662s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         	22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         	
 
_user_specified_nameinputs
 
W
;__inference_global_average_pooling1d_8_layer_call_fn_921772

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_920150i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
и
А
,__inference_embedding_8_layer_call_fn_921550

inputs
unknown:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_920336s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         	: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
┘
d
F__inference_dropout_35_layer_call_and_return_conditional_losses_922047

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Й
В
(__inference_model_8_layer_call_fn_921123
inputs_0
inputs_1
unknown:
	unknown_0:

	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identityИвStatefulPartitionedCallЮ
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
unknown_23*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_920483o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         	:         
: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
г	
д
G__inference_embedding_8_layer_call_and_return_conditional_losses_921560

inputs)
embedding_lookup_921554:
identityИвembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         	╗
embedding_lookupResourceGatherembedding_lookup_921554Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/921554*+
_output_shapes
:         	*
dtype0в
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/921554*+
_output_shapes
:         	Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         	w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:         	Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         	: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
┘
d
F__inference_dropout_35_layer_call_and_return_conditional_losses_920454

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ь
serving_default╪
=
input_171
serving_default_input_17:0         	
=
input_181
serving_default_input_18:0         
<
dense_350
StatefulPartitionedCall:0         tensorflow/serving/predict:╢а
▒
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
╡

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
ъ
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9_random_generator
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
е
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
╜
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~_random_generator
__call__
+А&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е_random_generator
Ж__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
л
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Оkernel
	Пbias
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
░
	Цiter
Чbeta_1
Шbeta_2

Щdecay
Ъlearning_ratemё#mЄ$mє-mЇ.mї=mЎ>mўTm°Um∙\m·]m√em№fm¤pm■qm 	ОmА	ПmБvВ#vГ$vД-vЕ.vЖ=vЗ>vИTvЙUvК\vЛ]vМevНfvОpvПqvР	ОvС	ПvТ"
	optimizer
р
0
#1
$2
%3
&4
-5
.6
=7
>8
?9
@10
T11
U12
\13
]14
e15
f16
g17
h18
p19
q20
r21
s22
О23
П24"
trackable_list_wrapper
а
0
#1
$2
-3
.4
=5
>6
T7
U8
\9
]10
e11
f12
p13
q14
О15
П16"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю2ы
(__inference_model_8_layer_call_fn_920536
(__inference_model_8_layer_call_fn_921123
(__inference_model_8_layer_call_fn_921179
(__inference_model_8_layer_call_fn_920921└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┌2╫
C__inference_model_8_layer_call_and_return_conditional_losses_921290
C__inference_model_8_layer_call_and_return_conditional_losses_921485
C__inference_model_8_layer_call_and_return_conditional_losses_920991
C__inference_model_8_layer_call_and_return_conditional_losses_921061└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╫B╘
!__inference__wrapped_model_919976input_17input_18"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
-
аserving_default"
signature_map
(:&2embedding_8/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
╓2╙
,__inference_embedding_8_layer_call_fn_921550в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_8_layer_call_and_return_conditional_losses_921560в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
*:(2batch_normalization_32/gamma
):'2batch_normalization_32/beta
2:0 (2"batch_normalization_32/moving_mean
6:4 (2&batch_normalization_32/moving_variance
<
#0
$1
%2
&3"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_32_layer_call_fn_921573
7__inference_batch_normalization_32_layer_call_fn_921586┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_921606
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_921640┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
!:
2dense_33/kernel
:2dense_33/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
╙2╨
)__inference_dense_33_layer_call_fn_921649в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_33_layer_call_and_return_conditional_losses_921660в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
5	variables
6trainable_variables
7regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_32_layer_call_fn_921665
+__inference_dropout_32_layer_call_fn_921670┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_32_layer_call_and_return_conditional_losses_921675
F__inference_dropout_32_layer_call_and_return_conditional_losses_921687┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
*:(2batch_normalization_34/gamma
):'2batch_normalization_34/beta
2:0 (2"batch_normalization_34/moving_mean
6:4 (2&batch_normalization_34/moving_variance
<
=0
>1
?2
@3"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_34_layer_call_fn_921700
7__inference_batch_normalization_34_layer_call_fn_921713┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_921733
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_921767┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
Є2я
;__inference_global_average_pooling1d_8_layer_call_fn_921772п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Н2К
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_921778п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_34_layer_call_fn_921783
+__inference_dropout_34_layer_call_fn_921788┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_34_layer_call_and_return_conditional_losses_921793
F__inference_dropout_34_layer_call_and_return_conditional_losses_921805┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
!:2dense_32/kernel
:2dense_32/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
╙2╨
)__inference_dense_32_layer_call_fn_921814в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_32_layer_call_and_return_conditional_losses_921825в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
!:2dense_34/kernel
:2dense_34/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╔non_trainable_variables
╩layers
╦metrics
 ╠layer_regularization_losses
═layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
╙2╨
)__inference_dense_34_layer_call_fn_921834в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_34_layer_call_and_return_conditional_losses_921845в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
*:(2batch_normalization_33/gamma
):'2batch_normalization_33/beta
2:0 (2"batch_normalization_33/moving_mean
6:4 (2&batch_normalization_33/moving_variance
<
e0
f1
g2
h3"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_33_layer_call_fn_921858
7__inference_batch_normalization_33_layer_call_fn_921871┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_921891
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_921925┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
*:(2batch_normalization_35/gamma
):'2batch_normalization_35/beta
2:0 (2"batch_normalization_35/moving_mean
6:4 (2&batch_normalization_35/moving_variance
<
p0
q1
r2
s3"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_35_layer_call_fn_921938
7__inference_batch_normalization_35_layer_call_fn_921951┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_921971
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_922005┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┤
╪non_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
z	variables
{trainable_variables
|regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_33_layer_call_fn_922010
+__inference_dropout_33_layer_call_fn_922015┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_33_layer_call_and_return_conditional_losses_922020
F__inference_dropout_33_layer_call_and_return_conditional_losses_922032┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_35_layer_call_fn_922037
+__inference_dropout_35_layer_call_fn_922042┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_35_layer_call_and_return_conditional_losses_922047
F__inference_dropout_35_layer_call_and_return_conditional_losses_922059┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
╪2╒
.__inference_concatenate_8_layer_call_fn_922065в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_concatenate_8_layer_call_and_return_conditional_losses_922072в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
!:2dense_35/kernel
:2dense_35/bias
0
О0
П1"
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
╙2╨
)__inference_dense_35_layer_call_fn_922081в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_35_layer_call_and_return_conditional_losses_922092в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
X
%0
&1
?2
@3
g4
h5
r6
s7"
trackable_list_wrapper
Ю
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
16"
trackable_list_wrapper
(
ь0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╘B╤
$__inference_signature_wrapper_921543input_17input_18"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
%0
&1"
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
?0
@1"
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
g0
h1"
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
r0
s1"
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

эtotal

юcount
я	variables
Ё	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
э0
ю1"
trackable_list_wrapper
.
я	variables"
_generic_user_object
-:+2Adam/embedding_8/embeddings/m
/:-2#Adam/batch_normalization_32/gamma/m
.:,2"Adam/batch_normalization_32/beta/m
&:$
2Adam/dense_33/kernel/m
 :2Adam/dense_33/bias/m
/:-2#Adam/batch_normalization_34/gamma/m
.:,2"Adam/batch_normalization_34/beta/m
&:$2Adam/dense_32/kernel/m
 :2Adam/dense_32/bias/m
&:$2Adam/dense_34/kernel/m
 :2Adam/dense_34/bias/m
/:-2#Adam/batch_normalization_33/gamma/m
.:,2"Adam/batch_normalization_33/beta/m
/:-2#Adam/batch_normalization_35/gamma/m
.:,2"Adam/batch_normalization_35/beta/m
&:$2Adam/dense_35/kernel/m
 :2Adam/dense_35/bias/m
-:+2Adam/embedding_8/embeddings/v
/:-2#Adam/batch_normalization_32/gamma/v
.:,2"Adam/batch_normalization_32/beta/v
&:$
2Adam/dense_33/kernel/v
 :2Adam/dense_33/bias/v
/:-2#Adam/batch_normalization_34/gamma/v
.:,2"Adam/batch_normalization_34/beta/v
&:$2Adam/dense_32/kernel/v
 :2Adam/dense_32/bias/v
&:$2Adam/dense_34/kernel/v
 :2Adam/dense_34/bias/v
/:-2#Adam/batch_normalization_33/gamma/v
.:,2"Adam/batch_normalization_33/beta/v
/:-2#Adam/batch_normalization_35/gamma/v
.:,2"Adam/batch_normalization_35/beta/v
&:$2Adam/dense_35/kernel/v
 :2Adam/dense_35/bias/v╘
!__inference__wrapped_model_919976о-.&#%$@=?>\]TUsprqhegfОПZвW
PвM
KЪH
"К
input_17         	
"К
input_18         

к "3к0
.
dense_35"К
dense_35         ╥
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_921606|&#%$@в=
6в3
-К*
inputs                  
p 
к "2в/
(К%
0                  
Ъ ╥
R__inference_batch_normalization_32_layer_call_and_return_conditional_losses_921640|%&#$@в=
6в3
-К*
inputs                  
p
к "2в/
(К%
0                  
Ъ к
7__inference_batch_normalization_32_layer_call_fn_921573o&#%$@в=
6в3
-К*
inputs                  
p 
к "%К"                  к
7__inference_batch_normalization_32_layer_call_fn_921586o%&#$@в=
6в3
-К*
inputs                  
p
к "%К"                  ╕
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_921891bhegf3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ ╕
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_921925bghef3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ Р
7__inference_batch_normalization_33_layer_call_fn_921858Uhegf3в0
)в&
 К
inputs         
p 
к "К         Р
7__inference_batch_normalization_33_layer_call_fn_921871Ughef3в0
)в&
 К
inputs         
p
к "К         ╕
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_921733b@=?>3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ ╕
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_921767b?@=>3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ Р
7__inference_batch_normalization_34_layer_call_fn_921700U@=?>3в0
)в&
 К
inputs         
p 
к "К         Р
7__inference_batch_normalization_34_layer_call_fn_921713U?@=>3в0
)в&
 К
inputs         
p
к "К         ╕
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_921971bsprq3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ ╕
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_922005brspq3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ Р
7__inference_batch_normalization_35_layer_call_fn_921938Usprq3в0
)в&
 К
inputs         
p 
к "К         Р
7__inference_batch_normalization_35_layer_call_fn_921951Urspq3в0
)в&
 К
inputs         
p
к "К         ╤
I__inference_concatenate_8_layer_call_and_return_conditional_losses_922072ГZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "%в"
К
0         
Ъ и
.__inference_concatenate_8_layer_call_fn_922065vZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "К         д
D__inference_dense_32_layer_call_and_return_conditional_losses_921825\TU/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ |
)__inference_dense_32_layer_call_fn_921814OTU/в,
%в"
 К
inputs         
к "К         д
D__inference_dense_33_layer_call_and_return_conditional_losses_921660\-./в,
%в"
 К
inputs         

к "%в"
К
0         
Ъ |
)__inference_dense_33_layer_call_fn_921649O-./в,
%в"
 К
inputs         

к "К         д
D__inference_dense_34_layer_call_and_return_conditional_losses_921845\\]/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ |
)__inference_dense_34_layer_call_fn_921834O\]/в,
%в"
 К
inputs         
к "К         ж
D__inference_dense_35_layer_call_and_return_conditional_losses_922092^ОП/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ ~
)__inference_dense_35_layer_call_fn_922081QОП/в,
%в"
 К
inputs         
к "К         о
F__inference_dropout_32_layer_call_and_return_conditional_losses_921675d7в4
-в*
$К!
inputs         	
p 
к ")в&
К
0         	
Ъ о
F__inference_dropout_32_layer_call_and_return_conditional_losses_921687d7в4
-в*
$К!
inputs         	
p
к ")в&
К
0         	
Ъ Ж
+__inference_dropout_32_layer_call_fn_921665W7в4
-в*
$К!
inputs         	
p 
к "К         	Ж
+__inference_dropout_32_layer_call_fn_921670W7в4
-в*
$К!
inputs         	
p
к "К         	ж
F__inference_dropout_33_layer_call_and_return_conditional_losses_922020\3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ ж
F__inference_dropout_33_layer_call_and_return_conditional_losses_922032\3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ ~
+__inference_dropout_33_layer_call_fn_922010O3в0
)в&
 К
inputs         
p 
к "К         ~
+__inference_dropout_33_layer_call_fn_922015O3в0
)в&
 К
inputs         
p
к "К         ж
F__inference_dropout_34_layer_call_and_return_conditional_losses_921793\3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ ж
F__inference_dropout_34_layer_call_and_return_conditional_losses_921805\3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ ~
+__inference_dropout_34_layer_call_fn_921783O3в0
)в&
 К
inputs         
p 
к "К         ~
+__inference_dropout_34_layer_call_fn_921788O3в0
)в&
 К
inputs         
p
к "К         ж
F__inference_dropout_35_layer_call_and_return_conditional_losses_922047\3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ ж
F__inference_dropout_35_layer_call_and_return_conditional_losses_922059\3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ ~
+__inference_dropout_35_layer_call_fn_922037O3в0
)в&
 К
inputs         
p 
к "К         ~
+__inference_dropout_35_layer_call_fn_922042O3в0
)в&
 К
inputs         
p
к "К         к
G__inference_embedding_8_layer_call_and_return_conditional_losses_921560_/в,
%в"
 К
inputs         	
к ")в&
К
0         	
Ъ В
,__inference_embedding_8_layer_call_fn_921550R/в,
%в"
 К
inputs         	
к "К         	╒
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_921778{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ н
;__inference_global_average_pooling1d_8_layer_call_fn_921772nIвF
?в<
6К3
inputs'                           

 
к "!К                  Ё
C__inference_model_8_layer_call_and_return_conditional_losses_920991и-.&#%$@=?>\]TUsprqhegfОПbв_
XвU
KЪH
"К
input_17         	
"К
input_18         

p 

 
к "%в"
К
0         
Ъ Ё
C__inference_model_8_layer_call_and_return_conditional_losses_921061и-.%&#$?@=>\]TUrspqghefОПbв_
XвU
KЪH
"К
input_17         	
"К
input_18         

p

 
к "%в"
К
0         
Ъ Ё
C__inference_model_8_layer_call_and_return_conditional_losses_921290и-.&#%$@=?>\]TUsprqhegfОПbв_
XвU
KЪH
"К
inputs/0         	
"К
inputs/1         

p 

 
к "%в"
К
0         
Ъ Ё
C__inference_model_8_layer_call_and_return_conditional_losses_921485и-.%&#$?@=>\]TUrspqghefОПbв_
XвU
KЪH
"К
inputs/0         	
"К
inputs/1         

p

 
к "%в"
К
0         
Ъ ╚
(__inference_model_8_layer_call_fn_920536Ы-.&#%$@=?>\]TUsprqhegfОПbв_
XвU
KЪH
"К
input_17         	
"К
input_18         

p 

 
к "К         ╚
(__inference_model_8_layer_call_fn_920921Ы-.%&#$?@=>\]TUrspqghefОПbв_
XвU
KЪH
"К
input_17         	
"К
input_18         

p

 
к "К         ╚
(__inference_model_8_layer_call_fn_921123Ы-.&#%$@=?>\]TUsprqhegfОПbв_
XвU
KЪH
"К
inputs/0         	
"К
inputs/1         

p 

 
к "К         ╚
(__inference_model_8_layer_call_fn_921179Ы-.%&#$?@=>\]TUrspqghefОПbв_
XвU
KЪH
"К
inputs/0         	
"К
inputs/1         

p

 
к "К         ъ
$__inference_signature_wrapper_921543┴-.&#%$@=?>\]TUsprqhegfОПmвj
в 
cк`
.
input_17"К
input_17         	
.
input_18"К
input_18         
"3к0
.
dense_35"К
dense_35         