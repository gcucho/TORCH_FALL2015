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
      if input:dim() == 3 then
        
        total_num_pixels = input:size(1)*input:size(2)*input:size(3)
  
        while counter< total_num_pixels/3 do
  
              --     if torch.bernoulli(.3) then
  
            k = math.ceil(torch.uniform(0,input:size(1)))
            i = math.ceil(torch.uniform(0,input:size(2)-3))
            j = math.ceil(torch.uniform(0,input:size(3)-3))
  
            if noise[{k,i,j}] >0 then
               --  noise[{k,{i,i+3},{j,j+3}}] = 0 
               noise[{k,i,{j,j+3}}] = 0 
            end
  
              counter = counter +1
                
         end
      elseif input:dim() == 2 then
        
        total_num_pixels = input:size(1)*input:size(2)
  
        kernel = 3

  
        while counter< total_num_pixels/math.ceil(kernel*1) do
  
              --     if torch.bernoulli(.3) then
                
            i = math.ceil(torch.uniform(1,input:size(1)-(kernel+1)))
            j = math.ceil(torch.uniform(1,input:size(2)-(kernel+1)))
	--print('input:size(1)' .. input:size(1))
	--print('input:size(2)' .. input:size(2))
	--print('i' .. i)
	--print('j' .. j)
		--i=1
		--j=1
		self.noise[{i,{j,j+kernel}}] = 0 
  	    if (j>(input:size(2)-(kernel+1))) then j = (input:size(2)-(kernel+1)) end
  	    if (i>(input:size(1)-(kernel+1))) then j = (input:size(1)-(kernel+1)) end
            if self.noise[{i,j}] > 0 then
--                 self.noise[{{i,i+kernel},{j,j+kernel}}] = 0 
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
