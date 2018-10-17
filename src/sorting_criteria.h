/** \file sorting_criteria.h
 * \brief  Varie funzioni per riordinare i nodi mettendo prima quelli dell'owner 0
*/

/** \brief a parita di owner, compara due nodi rispetto al grado totale (indegree+outdegree), prima grado inferiore
 */
int prima0poi1_alldeg(const void * a,const void * b) {
	const char *pval1 = (char *)(nodeOwner+*((int *)a));
	const char *pval2 = (char *)(nodeOwner+*((int *)b));
	const int *outdeg1 = (int *)(outDegrees_of_csr+*((int *)a));
	const int *outdeg2 = (int *)(outDegrees_of_csr+*((int *)b));
	const int *indeg1 = (int *)(inDegrees_of_csr+*((int *)a));
	const int *indeg2 = (int *)(inDegrees_of_csr+*((int *)b));
//printf(" CMP: %d %d  alldegs: %d %d\n",(*pval1) , (*pval2), ((*indeg1)+(*outdeg1)), ((*indeg2)+(*outdeg2)) );
	if ( (*pval1) == (*pval2) ) {
		if ( ((*indeg1)+(*outdeg1)) == ((*indeg2)+(*outdeg2)) ) {
			return(0);
		} else {
			return((((*indeg1)+(*outdeg1)) < ((*indeg2)+(*outdeg2))) ? -1 : 1);
		}
	} else {
		return(((*pval1) < (*pval2)) ? -1 : 1);
	}
}

/** \brief a parita di owner, compara due nodi rispetto a outdegree e a parita di outdegree compara con indegree 
 */
int prima0poi1_outdeg_indeg(const void * a,const void * b) {
	const char *pval1 = (char *)(nodeOwner+*((int *)a));
	const char *pval2 = (char *)(nodeOwner+*((int *)b));
	const int *outdeg1 = (int *)(outDegrees_of_csr+*((int *)a));
	const int *outdeg2 = (int *)(outDegrees_of_csr+*((int *)b));
	const int *indeg1 = (int *)(inDegrees_of_csr+*((int *)a));
	const int *indeg2 = (int *)(inDegrees_of_csr+*((int *)b));
//printf(" CMP: %d %d  outdegs: %d %d  intdegs: %d %d\n",(*pval1) , (*pval2), *outdeg1, *outdeg2 , *indeg1, *indeg2 );
	if ( (*pval1) == (*pval2) ) {
		if ( (*outdeg1) == (*outdeg2) ) {
			if ( (*indeg1) == (*indeg2) ) {
				return(0);
			} else {
				return(((*indeg1) < (*indeg2)) ? -1 : 1);
			}
		} else {
			return(((*outdeg1) < (*outdeg2)) ? -1 : 1);
		}
	} else {
		return(((*pval1) < (*pval2)) ? -1 : 1);
	}
}

/** \brief a parita di owner, compara due nodi rispetto a outdegree
 */
int prima0poi1_outdeg(const void * a,const void * b) {
	const char *pval1 = (char *)(nodeOwner+*((int *)a));
	const char *pval2 = (char *)(nodeOwner+*((int *)b));
	const int *outdeg1 = (int *)(outDegrees_of_csr+*((int *)a));
	const int *outdeg2 = (int *)(outDegrees_of_csr+*((int *)b));
//printf(" CMP: %d %d  outdegs: %d %d\n",(*pval1) , (*pval2), *outdeg1, *outdeg2 );
	if ( (*pval1) == (*pval2) ) {
		if ( (*outdeg1) == (*outdeg2) ) {
			return(0);
		} else {
			return(((*outdeg1) < (*outdeg2)) ? -1 : 1);
		}
	} else {
		return(((*pval1) < (*pval2)) ? -1 : 1);
	}
}

/** \brief a parita di owner, compara due nodi rispetto a indegree 
 */
int prima0poi1_indeg(const void * a,const void * b) {
	const char *pval1 = (char *)(nodeOwner+*((int *)a));
	const char *pval2 = (char *)(nodeOwner+*((int *)b));
	const int *indeg1 = (int *)(inDegrees_of_csr+*((int *)a));
	const int *indeg2 = (int *)(inDegrees_of_csr+*((int *)b));
//printf(" CMP: %d %d  intdegs: %d %d\n",(*pval1) , (*pval2), *indeg1, *indeg2 );
	if ( (*pval1) == (*pval2) ) {
		if ( (*indeg1) == (*indeg2) ) {
			return(0);
		} else {
			return(((*indeg1) < (*indeg2)) ? -1 : 1);
		}
	} else {
		return(((*pval1) < (*pval2)) ? -1 : 1);
	}
}

/** \brief compara due nodi solo rispetto all'owner: prima owner 0 e poi 1 
 */
int prima0poi1(const void * a,const void * b) {
        const char *pval1 = (char *)(nodeOwner+*((int *)a));
        const char *pval2 = (char *)(nodeOwner+*((int *)b));
//printf(" CMP: %d %d\n",(*pval1) , (*pval2) );
        if ( (*pval1) == (*pval2) ) 
                return(0);
        else 
                return(((*pval1) < (*pval2)) ? -1 : 1);
} 

