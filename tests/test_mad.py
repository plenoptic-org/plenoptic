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
        mad.synthesize(target, max_iter=10, loss_change_iter=5, store_progress=store_progress,
                       save_progress=store_progress, save_path=op.join(tmp_path, 'test_mad.pt'))
        if resume and store_progress:
            mad.synthesize(target, max_iter=10, loss_change_iter=5, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad.pt'), learning_rate=None,
                           initial_noise=None)
        mad.plot_synthesis_status()
        if store_progress:
            mad.animate()
        plt.close('all')

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
        mad.synthesize(target, max_iter=10, loss_change_iter=5, store_progress=store_progress,
                       save_progress=store_progress, save_path=op.join(tmp_path, 'test_mad.pt'))
        if resume and store_progress:
            mad.synthesize(target, max_iter=10, loss_change_iter=5, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad.pt'), learning_rate=None,
                           initial_noise=None)
        # for some reason, this causes the test to stall occasionally (I
        # think only with loss_func=range_penalty, target=model_1_max,
        # store_progress=False, resume=True). and trying to use
        # pytest-timeout doesn't work. it's not all that crucial, so
        # we'll get rid of it?
        mad.plot_synthesis_status()
        if store_progress:
            mad.animate()
        plt.close('all')

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
        mad.synthesize_all(max_iter=10, loss_change_iter=5, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad_{}.pt'))
        if resume and store_progress:
            mad.synthesize_all(resume, max_iter=10, loss_change_iter=5,
                               store_progress=store_progress,
                               save_progress=store_progress, learning_rate=None,
                               initial_noise=None, save_path=op.join(tmp_path, 'test_mad_{}.pt'))
        mad.plot_synthesized_image_all()
        mad.plot_loss_all()
        if store_progress:
            for t in ['model_1_min', 'model_2_min', 'model_1_max', 'model_2_max']:
                mad.animate(synthesis_target=t)
        plt.close('all')

    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    @pytest.mark.parametrize('model_name', ['V1', 'NLP', 'function'])
    @pytest.mark.parametrize('fraction_removed', [0, .1])
    @pytest.mark.parametrize('loss_change_fraction', [.5, 1])
    @pytest.mark.parametrize('coarse_to_fine', ['separate', 'together'])
    def test_coarse_to_fine(self, target, model_name, fraction_removed, loss_change_fraction,
                            coarse_to_fine, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        model2 = po.simul.models.naive.Identity()
        if model_name == 'V1':
            model1 = po.simul.PooledV1(1, img.shape[-2:]).to(DEVICE)
        elif model_name == 'NLP':
            model1 = po.metric.NLP().to(DEVICE)
        elif model_name == 'function':
            model1 = po.metric.nlpd
        mad = po.synth.MADCompetition(img, model1, model2)
        if model_name == 'V1' and 'model_1' in target:
            mad.synthesize(target, max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=coarse_to_fine, fraction_removed=fraction_removed,
                           loss_change_fraction=loss_change_fraction)
            mad.plot_synthesis_status()
            mad.save(op.join(tmp_path, 'test_mad_ctf.pt'))
            mad_copy = po.synth.MADCompetition.load(op.join(tmp_path, "test_mad_ctf.pt"),
                                                    map_location=DEVICE)
            # check the ctf-related attributes all saved correctly
            for k in ['coarse_to_fine', 'scales', 'scales_loss', 'scales_timing',
                      'scales_finished']:
                if not getattr(mad, k) == (getattr(mad_copy, k)):
                    raise Exception("Something went wrong with saving and loading! %s not the same"
                                    % k)
            mad_copy.synthesize(target, max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                                coarse_to_fine=coarse_to_fine, fraction_removed=fraction_removed,
                                loss_change_fraction=loss_change_fraction)
        else:
            # in this case, they'll first raise the exception that
            # metrics don't work with either of these
            if fraction_removed > 0 or loss_change_fraction < 1:
                exc = Exception
            # NLP and Identity have no scales attribute, and this
            # doesn't work with metrics either.
            else:
                exc = AttributeError
            with pytest.raises(exc):
                mad.synthesize(target, max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                               coarse_to_fine=coarse_to_fine, fraction_removed=fraction_removed,
                               loss_change_fraction=loss_change_fraction)
        plt.close('all')

    @pytest.mark.parametrize('model1', ['class', 'function'])
    @pytest.mark.parametrize('model2', ['class', 'function'])
    @pytest.mark.parametrize('loss_func', [None, 'l2', 'range_penalty_w_lmbda'])
    def test_save_load(self, model1, model2, loss_func, tmp_path):
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
        mad.synthesize('model_1_max', max_iter=10, loss_change_iter=5, store_progress=True)
        mad.save(op.join(tmp_path, 'test_mad_save_load.pt'))
        mad_copy = po.synth.MADCompetition.load(op.join(tmp_path, "test_mad_save_load.pt"),
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
        mad_copy.synthesize('model_1_max', max_iter=10, loss_change_iter=5, store_progress=True,
                            learning_rate=None, initial_noise=None)
        # and run another synthesis target (note neither learning_rate nor
        # initial_noise can be None in this case)
        mad_copy.synthesize('model_1_min', max_iter=10, loss_change_iter=5, store_progress=True)

    @pytest.mark.parametrize('model_name', ['class', 'function'])
    @pytest.mark.parametrize('fraction_removed', [0, .1])
    @pytest.mark.parametrize('loss_change_fraction', [.5, 1])
    def test_randomizers(self, model_name, fraction_removed, loss_change_fraction):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        model2 = po.simul.models.naive.Identity()
        if model_name == 'class':
            model1 = po.metric.NLP().to(DEVICE)
        elif model_name == 'function':
            model1 = po.metric.nlpd
        mad = po.synth.MADCompetition(img, model1, model2)
        if model_name == 'function' and (fraction_removed > 0 or loss_change_fraction < 1):
            with pytest.raises(Exception):
                mad.synthesize('model_1_min', max_iter=10, loss_change_iter=1,
                               loss_change_thresh=10, fraction_removed=fraction_removed,
                               loss_change_fraction=loss_change_fraction)
        else:
            mad.synthesize('model_1_min', max_iter=10, loss_change_iter=1,
                           fraction_removed=fraction_removed, loss_change_thresh=10,
                           loss_change_fraction=loss_change_fraction)
            mad.plot_synthesis_status()
        plt.close('all')

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
                mad.synthesize(target, max_iter=10, loss_change_iter=5,
                               learning_rate=learning_rate, initial_noise=initial_noise)
        else:
            mad.synthesize(target, max_iter=10, loss_change_iter=5, store_progress=False)
            if none_arg != 'lr':
                with pytest.raises(IndexError):
                    mad.synthesize(target, max_iter=10, loss_change_iter=5,
                                   learning_rate=learning_rate, initial_noise=initial_noise)
            else:
                # this actually will not raise an exception, because the
                # learning_rate has been initialized
                mad.synthesize(target, max_iter=10, loss_change_iter=5,
                               learning_rate=learning_rate, initial_noise=initial_noise)

    @pytest.mark.parametrize('optimizer', ['Adam', 'SGD', 'Adam-args'])
    @pytest.mark.parametrize('swa', [True, False])
    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    def test_optimizer_opts(self, optimizer, swa, target, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        model1 = po.simul.models.naive.Identity().to(DEVICE)
        model2 = po.metric.NLP().to(DEVICE)
        if '-' in optimizer:
            optimizer = optimizer.split('-')[0]
            optimizer_kwargs = {'weight_decay': .1}
        else:
            optimizer_kwargs = {}
        if swa:
            swa_kwargs = {'swa_start': 1, 'swa_freq': 1, 'swa_lr': .05}
        else:
            swa_kwargs = {}
        mad = po.synth.MADCompetition(img, model1, model2)
        mad.synthesize(target, max_iter=10, loss_change_iter=5, swa=swa, swa_kwargs=swa_kwargs,
                       optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)

