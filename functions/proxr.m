function H = proxr( Hb, ops, alpha, d)
    switch ops.constraint{d}
        case 'nn' %'nn' for nonegative;
            H =  max(0 , Hb) ;
        case 'simplex_col' % for simplex constraints with each column
            H = ProjectOntoSimplex(Hb, ops.lambda);
        case 'simplex_row' % for simplex constraints with each row
            H = ProjectOntoSimplex(Hb', 1);
            H = H';
        case 'l1' % for L1 norm; 
            H = sign( Hb ).* max( 0, abs(Hb) - ops.lambda*alpha);
        case 'l1n' % for L1 norm + nonegative
            H = max( 0, abs(Hb) - alpha*ops.lambda );
        case 'l2' % for L2 norm 
            H =  Hb/(1+2*alpha*ops.lambda);
        case 'l2n' % for L2 norm + nonegative
            H = max(0,Hb)/(1+2*alpha*ops.lambda);
        case 'l2-bound' % for ball constraints
           nn = sqrt( sum( Hb.^2 ) );
            H = Hb * diag( 1./ max(1,nn) );
        case 'l2-boundn'  % for ball constraints + nonnegative
            H = max( 0, Hb );
           nn = sqrt( sum( H.^2 ) );
            H = H * diag( 1./ max(1,nn) );
        case 'l0' % for L0 norm
            T = sort(Hb,2,'descend');
            t = T(:,4); T = repmat(t,1,size(T,2));
            H = Hb .* ( Hb >= T );
         case 'nc' % for no constraint
            H = Hb;
    end
end