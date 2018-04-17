clc; clear; close all

% Q = diag([501,501,1,1]);
% R = 1;
% u = rand(1001,1)-0.5;
% 
% Wu_ini = [rand(1,4)-0.5, zeros(1,14)]';
% Wv_ini = zeros(10,1);
% Wu     = Wu_ini;
% Wv     = Wv_ini;
% WU     = Wu;
% WV     = Wv;
% 
% for j=1:20
%     EE=[];
%     PP=[];
%     x = [-1 1 0 .5]';   X = x;  k = 1;
%     for i = 0.1:0.1:100
%         f =@(t,x)[x(2);-0.5*(x(1) + x(2) - x(3) + x(4)) + 0.5*(x(1) + x(3))^2*(x(2) + x(4)) + u(k); x(4);-x(3)];
%         [t,xx]  = ode45(f,[i - 0.1, i],x);
%         [m,n]   = size(xx);
%         x_next  = xx(m,:)';
%         P1 = 1.0005*Cri_NN(x) - 0.9995*Cri_NN(x_next);
%         P2 = 0.05*(x'*Q*x + Act_NN(x)'*Wu*R*Act_NN(x)'*Wu + x_next'*Q*x_next + Act_NN(x_next)'*Wu*R*Act_NN(x_next)'*Wu);
%         P3 = 0.05*R*((Act_NN(x)'*Wu - u(k))*Act_NN(x)' + (Act_NN(x_next)'*Wu - u(k+1))*Act_NN(x_next)');
%         
%         P  = [P1',2*P3];
%         E  = P2;
%         PP = [PP;P];
%         EE = [EE;E];
%         x       = x_next;
%         X       = [X,x_next];
%         k       = k+1;
%     end
%     W = inv(PP'*PP)*PP'*EE;
%     norm(W - [Wv;Wu])
%     if norm(W - [Wv;Wu])<1e-5
%         Wv = W(1:10);
%         Wu = W(11:28);
%         WU = [WU,Wu];
%         WV = [WV,Wv];
%         Iteration = j
%         break;
%     end
%     Wv = W(1:10);
%     Wu = W(11:28);
%     WU = [WU,Wu];
%     WV = [WV,Wv];
%     Iteration = j;
% end

 load('optimalResults_2.mat')


figure(1); plot(0 : Iteration, WU);ylabel('$\hat{\theta}_u^{(i)}$','Interpreter','latex');xlabel('$Iteration \quad i$','Interpreter','latex')
figure(2); plot(0 : Iteration, WV);ylabel('$\hat{\theta}_V^{(i)}$','Interpreter','latex');xlabel('$Iteration \quad i$','Interpreter','latex')
for s = 0: Iteration
    WWU(s+1) = norm(WU(:,s+1));
    WWV(s+1) = norm(WV(:,s+1));
end
figure(11);plot(0 : Iteration, WWU, '-o');ylabel('$\Vert \hat{\theta}_u^{(i)} \Vert$','Interpreter','latex');xlabel('$Iteration \quad i$','Interpreter','latex')
figure(22);plot(0 : Iteration, WWV, '-s');ylabel('$\Vert \hat{\theta}_V^{(i)} \Vert$','Interpreter','latex');xlabel('$Iteration \quad i$','Interpreter','latex')

x = [-1 1 0 .5]';
f =@(t,x)[x(2);-0.5*(x(1) + x(2) - x(3) + x(4)) + 0.5*(x(1) + x(3))^2*(x(2) + x(4)) + Act_NN(x)'*Wu+rand(1,1)*cos(x(2)+x(4))*sin(x(1)+x(3)); x(4);-x(3)];
[t,xx]   = ode45(f,[0, 30],x);
[t,xx]   = ode45(f,[0, 30],x);
[m,n]    = size(t);
u =0;
for i = 1 : m
    u(i) = Act_NN(xx(i,:))'*Wu;
end
figure(13); plot(t, xx(:,1:2:3));legend('x_1','x_3');xlabel('$Time$','Interpreter','latex')
figure(24); plot(t, xx(:,2:2:4));legend('x_2','x_4');xlabel('$Time$','Interpreter','latex')
figure(5);  plot(t, u);ylabel('$\hat{u}^{(14)}$','Interpreter','latex');xlabel('$Time$','Interpreter','latex')
figure(132); plot(t, xx(:,1)+xx(:,3),t,xx(:,3),'--');legend('x_1','r_1');xlabel('$Time$','Interpreter','latex')
figure(242); plot(t, xx(:,2)+xx(:,4),t,xx(:,4),'--');legend('x_2','r_2');xlabel('$Time$','Interpreter','latex')


x = [-1 1 0 .5]';
for i=0:Iteration
    h(i+1) = Cri_NN(x)'*WV(:,i+1);
end
figure(111);plot(1:Iteration,h(2:Iteration+1),'-d');
xlabel('$Iteration \quad i$','Interpreter','latex');
ylabel('$\hat{V}^{(i)}$','Interpreter','latex')
axis([1,Iteration,0, max(h)])
