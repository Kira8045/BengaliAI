def train(self, epoch, no_of_steps, trainloader, lr):
        self.model.train()

        train_loss, correct, total = 0, 0, 0

        # Declare optimizer.
        if not hasattr(self, 'optimizer'):
            if self.fp16_mode:
                self.optimizer = optim.Adam(
                    self.master_params, lr)
            else:
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr,)

        # If epoch less than 5 use warmup, else use scheduler.
        if epoch < 5:
            lr = self.warmup_learning_rate(lr, no_of_steps, epoch,
                                           len(trainloader))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == 5:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( self.optimizer, mode="min", 
                                                            factor = 0.3,
                                                            patience = 5,
                                                            verbose = False)
        if epoch >= 5:
            scheduler.step(epoch=epoch)

        print('Learning Rate: %g' % (list(
            map(lambda group: group['lr'], self.optimizer.param_groups)))[0])
        # Loss criterion is in FP32.

        for idx, d in enumerate(trainloader):
            inputs = d["image"]
            grapheme_root = d["grapheme_root"]
            vowel_diacritic = d["vowel_diacritic"]
            consonant_diacritic = d["consonant_diacritic"]

            if self.train_on_gpu:
                inputs, grapheme_root,vowel_diacritic,consonant_diacritic = inputs.cuda(), grapheme_root.cuda(), vowel_diacritic.cuda(),consonant_diacritic.cuda()

            self.model.zero_grad()
            outputs = self.model(inputs)
            # We calculate the loss in FP32 since reduction ops can be
            # wrong when represented in FP16.
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            
            if self.loss_scaling:
                # Sometime the loss may become small to be represente in FP16
                # So we scale the losses by a large power of 2, 2**7 here.
                loss = loss * self._LOSS_SCALE
            # Calculate the gradients
            loss.backward()
            if self.fp16_mode:
                # Now we move the calculated gradients to the master params
                # so that we can apply the gradient update in FP32.
                self.model_grads_to_master_grads(self.model_params,
                                                 self.master_params)
                if self.loss_scaling:
                    # If we scaled our losses now is a good time to scale it
                    # back since our gradients are in FP32.
                    for params in self.master_params:
                        params.grad.data = params.grad.data / self._LOSS_SCALE
                # Apply weight update in FP32.
                self.optimizer.step()
                # Copy the updated weights back FP16 model weights.
                self.master_params_to_model_params(self.model_params,
                                                   self.master_params)
            else:
                self.optimizer.step()