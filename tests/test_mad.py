# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
import matplotlib
matplotlib.use('agg')
import pytest
import matplotlib.pyplot as plt
import plenoptic as po
import torch
import os.path as op
from test_plenoptic import DEVICE, DATA_DIR


class TestMAD(object):

    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    @pytest.mark.parametrize('model1', ['class', 'function'])
    @pytest.mark.parametrize('model2', ['class', 'function'])
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    @pytest.mark.parametrize('resume', [False, True])
    def test_basic(self, target, model1, model2, store_progress, resume, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        if model1 == 'class':
            model1 = po.simul.models.naive.Identity().to(DEVICE)
        elif model1 == 'function':
            model1 = po.metric.naive.mse
        if model2 == 'class':
            model2 = po.metric.NLP().to(DEVICE)
        elif model2 == 'function':
            model2 = po.metric.nlpd
        mad = po.synth.MADCompetition(img, model1, model2)
        mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=store_progress,
                       save_progress=store_progress, save_path=op.join(tmp_path, 'test_mad.pt'))
        if resume and store_progress:
            mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad.pt'), learning_rate=None,
                           initial_noise=None)

    @pytest.mark.parametrize('loss_func', [None, 'l2', 'mse', 'range_penalty',
                                           'range_penalty_w_lmbda',
                                           'mse_range_penalty_w_lmbda'])
    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    @pytest.mark.parametrize('resume', [False, True])
    def test_loss_func(self, loss_func, target, store_progress, resume, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        model1 = po.simul.models.naive.Identity().to(DEVICE)
        model2 = po.metric.NLP().to(DEVICE)
        loss_kwargs = {}
        if loss_func is None:
            loss = None
        elif loss_func == 'l2':
            loss = po.optim.l2_norm
        elif loss_func == 'mse':
            loss = po.optim.mse
        elif loss_func == 'range_penalty':
            loss = po.optim.l2_and_penalize_range
        elif loss_func == 'range_penalty_w_lmbda':
            loss = po.optim.l2_and_penalize_range
            loss_kwargs['lmbda'] = .1
        elif loss_func == 'mse_range_penalty_w_lmbda':
            loss = po.optim.mse_and_penalize_range
            loss_kwargs['lmbda'] = .1
        mad = po.synth.MADCompetition(img, model1, model2, loss_function=loss,
                                      loss_function_kwargs=loss_kwargs)
        mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=store_progress,
                       save_progress=store_progress, save_path=op.join(tmp_path, 'test_mad.pt'))
        if resume and store_progress:
            mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad.pt'), learning_rate=None,
                           initial_noise=None)

    @pytest.mark.parametrize('model1', ['class', 'function'])
    @pytest.mark.parametrize('model2', ['class', 'function'])
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    @pytest.mark.parametrize('resume', [False, 'skip', 're-run', 'continue'])
    def test_all(self, model1, model2, store_progress, resume, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        if model1 == 'class':
            model1 = po.simul.models.naive.Identity().to(DEVICE)
        else:
            model1 = po.metric.naive.mse
        if model2 == 'class':
            model2 = po.metric.NLP().to(DEVICE)
        else:
            model2 = po.metric.nlpd
        mad = po.synth.MADCompetition(img, model1, model2)
        mad.synthesize_all(max_iter=5, loss_change_iter=3, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad_{}.pt'))
        if resume and store_progress:
            mad.synthesize_all(resume, max_iter=5, loss_change_iter=3,
                               store_progress=store_progress,
                               save_progress=store_progress, learning_rate=None,
                               initial_noise=None, save_path=op.join(tmp_path, 'test_mad_{}.pt'))
        mad.plot_synthesized_image_all()
        mad.plot_loss_all()
        if store_progress:
            for t in ['model_1_min', 'model_2_min', 'model_1_max', 'model_2_max']:
                mad.animate(synthesis_target=t).to_html5_video()
        plt.close('all')

    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    @pytest.mark.parametrize('model_name', ['SPyr', 'NLP', 'function'])
    @pytest.mark.parametrize('coarse_to_fine', ['separate', 'together'])
    def test_coarse_to_fine(self, target, model_name, coarse_to_fine, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        model2 = po.simul.models.naive.Identity()
        if model_name == 'SPyr':
            # with downsample=False, we get a tensor back. setting height=1 and
            # order=1 limits the size
            model1 = po.simul.Steerable_Pyramid_Freq(img.shape[-2:], downsample=False,
                                                     height=1, order=1).to(DEVICE)
        elif model_name == 'NLP':
            model1 = po.metric.NLP().to(DEVICE)
        elif model_name == 'function':
            model1 = po.metric.nlpd
        mad = po.synth.MADCompetition(img, model1, model2)
        if model_name == 'SPyr' and 'model_1' in target:
            mad.synthesize(target, max_iter=5, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=coarse_to_fine)
            assert len(mad.scales_finished) > 0, "Didn't actually switch scales!"
            mad.save(op.join(tmp_path, 'test_mad_ctf.pt'))
            mad_copy = po.synth.MADCompetition(img, model1, model2)
            mad_copy.load(op.join(tmp_path, "test_mad_ctf.pt"),
                          map_location=DEVICE)
            # check the ctf-related attributes all saved correctly
            for k in ['coarse_to_fine', 'scales', 'scales_loss', 'scales_timing',
                      'scales_finished']:
                if not getattr(mad, k) == (getattr(mad_copy, k)):
                    raise Exception("Something went wrong with saving and loading! %s not the same"
                                    % k)
            mad_copy.synthesize(target, max_iter=5, loss_change_iter=1, loss_change_thresh=10,
                                coarse_to_fine=coarse_to_fine)
        else:
            with pytest.raises(AttributeError):
                mad.synthesize(target, max_iter=5, loss_change_iter=1, loss_change_thresh=10,
                               coarse_to_fine=coarse_to_fine)

    @pytest.mark.parametrize('model1', ['class', 'function'])
    @pytest.mark.parametrize('model2', ['class', 'function'])
    @pytest.mark.parametrize('loss_func', [None, 'l2', 'range_penalty_w_lmbda'])
    @pytest.mark.parametrize('fail', [False, 'img', 'model1', 'model2', 'loss'])
    def test_save_load(self, model1, model2, loss_func, fail, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm'))
        if model1 == 'class':
            model1 = po.simul.models.naive.Identity().to(DEVICE)
        else:
            model1 = po.metric.naive.mse
        if model2 == 'class':
            model2 = po.metric.NLP().to(DEVICE)
        else:
            model2 = po.metric.nlpd
        loss_kwargs = {}
        if loss_func is None:
            loss = None
        elif loss_func == 'l2':
            loss = po.optim.l2_norm
        elif loss_func == 'range_penalty_w_lmbda':
            loss = po.optim.l2_and_penalize_range
            loss_kwargs['lmbda'] = .1
        mad = po.synth.MADCompetition(img, model1, model2, loss_function=loss,
                                      loss_function_kwargs=loss_kwargs)
        mad.synthesize('model_1_max', max_iter=5, loss_change_iter=3, store_progress=True)
        mad.save(op.join(tmp_path, 'test_mad_save_load.pt'))
        # when the model is a function, the loss_function is ignored and thus
        # we won't actually fail to load here (we check against the specific
        # callable because we've overwritten the model input arg)
        if fail and not (fail == 'loss' and model1 == po.metric.mse and model2 == po.metric.nlpd):
            if fail == 'img':
                img = torch.rand_like(img)
            elif fail == 'model1':
                model1 = po.metric.nlpd
            elif fail == 'model2':
                model2 = po.metric.mse
            elif fail == 'loss':
                loss = lambda *args, **kwargs: 1
                loss_kwargs = {}
            mad_copy = po.synth.MADCompetition(img, model1, model2, loss_function=loss,
                                               loss_function_kwargs=loss_kwargs)
            with pytest.raises(Exception):
                mad_copy.load(op.join(tmp_path, "test_mad_save_load.pt"),
                              map_location=DEVICE)
        else:
            mad_copy = po.synth.MADCompetition(img, model1, model2, loss_function=loss,
                                               loss_function_kwargs=loss_kwargs)
            mad_copy.load(op.join(tmp_path, "test_mad_save_load.pt"),
                          map_location=DEVICE)
            if mad.synthesis_target != mad_copy.synthesis_target:
                raise Exception("Something went wrong with saving and loading! synthesis_"
                                "target not the same!")
            for k in mad_copy._attrs_all:
                orig = getattr(mad, k+"_all")
                saved = getattr(mad_copy, k+"_all")
                for ki, v in orig.items():
                    eql = False
                    try:
                        if v == saved[ki]:
                            eql = True
                    except RuntimeError:
                        # then it's a tensor
                        if v.allclose(saved[ki]):
                            eql = True
                    if not eql:
                        raise Exception(f"Something went wrong with saving and loading! {k, ki} not the same!")
                eql = False
                try:
                    if saved[mad.synthesis_target] == getattr(mad_copy, k):
                        eql = True
                except RuntimeError:
                    # then it's a tensor
                    if saved[mad.synthesis_target].allclose(getattr(mad_copy, k)):
                        eql = True
                if not eql:
                    raise Exception(f"Something went wrong with saving and loading! {k} and its _all"
                                    "version not properly synced")
            # check these attributes all saved correctly
            for k in ['base_signal', 'saved_representation_1', 'saved_signal',
                      'synthesized_representation_1', 'synthesized_signal', 'base_representation_1',
                      'saved_representation_2', 'synthesized_representation_2', 'base_representation_2']:
                if not getattr(mad, k).allclose(getattr(mad_copy, k)):
                    raise Exception("Something went wrong with saving and loading! %s not the same"
                                    % k)
            assert not isinstance(mad_copy.synthesized_representation, torch.nn.Parameter), "synthesized_rep shouldn't be a parameter!"
            # check loss functions correctly saved
            mad_loss = mad.loss_function_1(mad.synthesized_representation_1, mad.base_representation_1,
                                           mad.synthesized_signal, mad.base_signal)
            mad_copy_loss = mad_copy.loss_function_1(mad_copy.synthesized_representation_1,
                                                     mad_copy.base_representation_1,
                                                     mad_copy.synthesized_signal, mad_copy.base_signal)
            if mad_loss != mad_copy_loss:
                raise Exception(f"Loss function 1 not properly saved! Before saving was {mad_loss}, "
                                f"after loading was {mad_copy_loss}")
            mad_loss = mad.loss_function_2(mad.synthesized_representation_2, mad.base_representation_2,
                                           mad.synthesized_signal, mad.base_signal)
            mad_copy_loss = mad_copy.loss_function_2(mad_copy.synthesized_representation_2,
                                                     mad_copy.base_representation_2,
                                                     mad_copy.synthesized_signal, mad_copy.base_signal)
            if mad_loss != mad_copy_loss:
                raise Exception(f"Loss function 2 not properly saved! Before saving was {mad_loss}, "
                                f"after loading was {mad_copy_loss}")
            # check that can resume
            mad_copy.synthesize('model_1_max', max_iter=5, loss_change_iter=3, store_progress=True,
                                learning_rate=None, initial_noise=None)
            # and run another synthesis target (note neither learning_rate nor
            # initial_noise can be None in this case)
            mad_copy.synthesize('model_1_min', max_iter=5, loss_change_iter=3, store_progress=True)

    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    @pytest.mark.parametrize('model1', ['class', 'function'])
    @pytest.mark.parametrize('model2', ['class', 'function'])
    @pytest.mark.parametrize('none_arg', ['lr', 'initial_noise', 'both'])
    @pytest.mark.parametrize('none_place', ['before', 'after'])
    def test_resume_exceptions(self, target, model1, model2, none_arg, none_place):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        if model1 == 'class':
            model1 = po.simul.models.naive.Identity().to(DEVICE)
        elif model1 == 'function':
            model1 = po.metric.naive.mse
        if model2 == 'class':
            model2 = po.metric.NLP().to(DEVICE)
        elif model2 == 'function':
            model2 = po.metric.nlpd
        mad = po.synth.MADCompetition(img, model1, model2)
        learning_rate = 1
        initial_noise = .1
        if none_arg == 'lr' or none_arg == 'both':
            learning_rate = None
        if none_arg == 'initial_noise' or none_arg == 'both':
            initial_noise = None
        # can't call synthesize() with initial_noise=None or
        # learning_rate=None unless synthesize() has been called before
        # with store_progress!=False
        if none_place == 'before':
            with pytest.raises(IndexError):
                mad.synthesize(target, max_iter=5, loss_change_iter=3,
                               learning_rate=learning_rate, initial_noise=initial_noise)
        else:
            mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=False)
            if none_arg != 'lr':
                with pytest.raises(IndexError):
                    mad.synthesize(target, max_iter=5, loss_change_iter=3,
                                   learning_rate=learning_rate, initial_noise=initial_noise)
            else:
                # this actually will not raise an exception, because the
                # learning_rate has been initialized
                mad.synthesize(target, max_iter=5, loss_change_iter=3,
                               learning_rate=learning_rate, initial_noise=initial_noise)

    @pytest.mark.parametrize('optimizer', ['Adam', 'SGD', 'Adam-args'])
    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    def test_optimizer_opts(self, optimizer,target, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        model1 = po.simul.models.naive.Identity().to(DEVICE)
        model2 = po.metric.NLP().to(DEVICE)
        if '-' in optimizer:
            optimizer = optimizer.split('-')[0]
            optimizer_kwargs = {'weight_decay': .1}
        else:
            optimizer_kwargs = {}
        mad = po.synth.MADCompetition(img, model1, model2)
        mad.synthesize(target, max_iter=5, loss_change_iter=3,
                       optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)
