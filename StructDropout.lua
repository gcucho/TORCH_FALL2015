local StructDropout, Parent = torch.class('nn.StructDropout', 'nn.Module')

function StructDropout:__init(p,v1)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   -- version 2 scales output during training instead of evaluation
   self.v2 = not v1
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end

function StructDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      self.noise:resizeAs(input)
      self.noise:fill(1)
      counter=0
--print('input:dim()' .. input:dim())
--print('input:size(1)' .. input:size(1))
--print('input:size(2)' .. input:size(2))
--print('input:size(3)' .. input:size(3))
--print('input:size(4)' .. input:size(4))
	NumDim = input:dim();
      if input:dim() == 4 then	
	total_num_pixels = input:size(1)*input:size(2)*input:size(3)*input:size(4)
	kernel = math.ceil(math.min(input:size(NumDim-1),input:size(NumDim))*0.1)-1
	--print('kernel:',kernel)
	--kernel = 2;
	
	--affecting only 3 last dimensions k,i,j
	
        --mean_i = math.ceil(input:size(2)/2)
        --mean_j = math.ceil(input:size(3)/2)
        --mean_k = math.ceil(input:size(4)/2)
	--std_i = math.ceil(input:size(2)/5)
	--std_j = math.ceil(input:size(3)/3)
	--std_k = math.ceil(input:size(4)/3)

	while counter< total_num_pixels/1000 do --Only 1/3 of elements will be changed
	-- Normal distribuition --
		--i = math.ceil(mean_i + std_i*torch.randn(1))
        	  --  if i<=0 then i=1 end
	          --  if i>(input:size(2)-kernel) then i=(input:size(2)-kernel) end
		--j = math.ceil(torch.normal(mean_j,std_j))
	          --  if j<=0 then j=1 end
        	  --  if j>(input:size(3)-kernel) then j=(input:size(3)-kernel) end
		--k = math.ceil(torch.normal(mean_k,std_k))
        	  --  if k<=0 then k=1 end
	          --  if k>(input:size(4)-kernel) then k=(input:size(4)-kernel) end
	-- Uniform distribuition --
		batch_sd = math.ceil(torch.uniform(1,input:size(1)))
		i = math.ceil(torch.uniform(1,input:size(2)-kernel))
	        j = math.ceil(torch.uniform(1,input:size(3)-kernel))
	        k = math.ceil(torch.uniform(1,input:size(4)-kernel))
		--print( batch_sd, i, j, k)
		--print(self.noise:size(1))
		if self.noise[{batch_sd,i,j,k}] > 0 then
	               --noise[{k,{i,i+3},{j,j+3}}] = 0 
	           self.noise[{batch_sd,i,{j,j+kernel},{k,k+kernel}}] = 0 
	        end
        	counter = counter +1
         end

      elseif input:dim() == 3 then
  
        total_num_pixels = input:size(1)*input:size(2)*input:size(3)
	kernel = math.ceil(math.min(input:size(NumDim-1),input:size(NumDim))*0.1)-1
	--print('kernel:',kernel)
	--kernel = 2;
		
 	while counter< total_num_pixels/10 do --Only 1/10 of elements will be changed

	-- Uniform distribuition -	
		i = math.ceil(torch.uniform(1,input:size(1)))
	        j = math.ceil(torch.uniform(1,input:size(2)-kernel))
	        k = math.ceil(torch.uniform(1,input:size(3)-kernel))

		if self.noise[{i,j,k}] > 0 then
	           self.noise[{i,{j,j+kernel},{k,k+kernel}}] = 0 
	        end
        	counter = counter +1
         end
      elseif input:dim() == 2 then
        
        total_num_pixels = input:size(1)*input:size(2)
	kernel = math.ceil(math.min(input:size(NumDim-1),input:size(NumDim))*0.1)-1
	--print('kernel:',kernel)
	--kernel = 2;
		
 	while counter< total_num_pixels/10 do --Only 1/10 of elements will be changed

	-- Uniform distribuition -	
		i = math.ceil(torch.uniform(1,input:size(1)))
	        j = math.ceil(torch.uniform(1,input:size(2)-kernel))

		if self.noise[{i,j}] > 0 then
	           self.noise[{i,{j,j+kernel}}] = 0 
	        end
        	counter = counter +1
         end
--        self.noise:bernoulli(1-self.p)
      end
        
      if self.v2 then
         self.noise:div(1-self.p)
      end
      self.output:cmul(self.noise)
   elseif not self.v2 then
      self.output:mul(1-self.p)
   end
   return self.output
end

function StructDropout:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function StructDropout:setp(p)
   self.p = p
end

function StructDropout:__tostring__()
  return string.format('%s(%f)', torch.type(self), self.p)
end
