
def Vector( vector, inicio, final	 ):
	Salida = range(MAX)

	ind = 0

	for i in range(inicio, final):						# rellenando los vectores individuales
		Salida[ind] = vector[i]
		ind += 1

	return Salida


def Comparaciones( f1, f2, MAX):     

	Salida = range(MAX)

	for i in range(MAX):								# multiplicando elementos de vectores
		Salida[i] = f1[i] * f2[i]

	return Salida
	


# ... PROCESOS anteriores ... 

MAX = 60001
batch = 10
	
V1 = range(60001*10)									# vector que representa vector final de fft
V2 = range(60001*10)									# vector que representa vector final de complejo conjugado
	
inicio = 0
final = MAX
pasos = 0

n = 1 


cont = 0
for i in range(batch-1):								# 45 comparaciones,  num. comparaciones definidas por -->  ((batch)*(batch-1))/2 
	s1 = Vector( V1, inicio, final)						# creando vector individual a partir del vector de final de fft
	inicio2 = MAX + pasos
	final2 = (MAX+MAX) + pasos

	for i in range(batch-n):
		s2 = Vector( V2, inicio2, final2 )				# creando el vector individual apartir del vector final de complejo conjugado
		f1xf2 = Comparaciones( s1, s2, MAX )			# el vector salida se le aplica ifft

		inicio2 += MAX									# avanzando cada 60001 elementos
		final2 += MAX

# ... Salida f1xf2 rumbo a ifft ...            dentro de doble for

	batch -= 1
	inicio += MAX
	final += MAX
	pasos += MAX


